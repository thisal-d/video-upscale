#include "io/encoder.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>

#include "core/log.hpp"
#include "core/system.hpp"
#include "io/ffmpeg_utils.hpp"

namespace vu {

struct VideoEncoder::Impl {
  AVFormatContext* oc = nullptr;
  ff::CodecCtxPtr enc;
  ff::AVFramePtr frame;
  ff::AVPacketPtr pkt;
  AVStream* vst = nullptr;
  std::vector<int> inToOut;                 // input stream index -> output stream index (-1: dropped)
  std::vector<AVRational> inTimeBase;
  std::mutex muxMutex;
  bool headerWritten = false;
  bool finished = false;
  EncoderStats stats;
  std::string path;
  AVRational srcTb{1, 25};

  ~Impl() {
    if (oc) {
      if (headerWritten && !finished) {
        // Abnormal termination: still try to leave a playable file.
        av_write_trailer(oc);
      }
      if (!(oc->oformat->flags & AVFMT_NOFILE)) avio_closep(&oc->pb);
      avformat_free_context(oc);
    }
  }

  void writeMuxed(AVPacket* p) {
    std::lock_guard lk(muxMutex);
    int ret = av_interleaved_write_frame(oc, p);
    if (ret < 0) fail(ErrorCode::OutputError, "muxing failed: " + ff::errStr(ret));
  }

  void drain() {
    while (true) {
      int ret = avcodec_receive_packet(enc.get(), pkt.get());
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) return;
      ff::check(ret, ErrorCode::OutputError, "video encode failed");
      av_packet_rescale_ts(pkt.get(), enc->time_base, vst->time_base);
      pkt->stream_index = vst->index;
      stats.videoBytes += pkt->size;
      writeMuxed(pkt.get());
    }
  }
};

VideoEncoder::VideoEncoder() : d_(std::make_unique<Impl>()) {}
VideoEncoder::~VideoEncoder() = default;
bool VideoEncoder::isOpen() const { return d_->oc != nullptr; }
EncoderStats VideoEncoder::stats() const { return d_->stats; }

namespace {

const AVCodec* pickEncoder(const std::string& name) {
  const AVCodec* c = avcodec_find_encoder_by_name(name.c_str());
  if (c) return c;
  log::warn("encoder '{}' not available in this FFmpeg build", name);
  for (const char* alt : {"libx264", "libx265", "libopenh264", "mpeg4"}) {
    if (name == alt) continue;
    if ((c = avcodec_find_encoder_by_name(alt))) {
      log::warn("falling back to encoder '{}'", alt);
      return c;
    }
  }
  return nullptr;
}

void inferColor(ColorInfo& c, int srcHeight) {
  // Unspecified color metadata is interpreted by players from the *output* resolution
  // (SD -> BT.601, HD -> BT.709). Since upscaling changes the resolution class, we tag the
  // output explicitly with what the source most likely used so colors do not shift.
  const bool sd = srcHeight > 0 && srcHeight <= 576;
  if (c.space == AVCOL_SPC_UNSPECIFIED) c.space = sd ? (srcHeight == 576 ? AVCOL_SPC_BT470BG : AVCOL_SPC_SMPTE170M) : AVCOL_SPC_BT709;
  if (c.primaries == AVCOL_PRI_UNSPECIFIED) c.primaries = sd ? (srcHeight == 576 ? AVCOL_PRI_BT470BG : AVCOL_PRI_SMPTE170M) : AVCOL_PRI_BT709;
  if (c.transfer == AVCOL_TRC_UNSPECIFIED) c.transfer = sd ? AVCOL_TRC_SMPTE170M : AVCOL_TRC_BT709;
  if (c.range == AVCOL_RANGE_UNSPECIFIED) c.range = AVCOL_RANGE_MPEG;
}

}  // namespace

void VideoEncoder::open(const std::string& path, const EncoderSettings& s, const AVFormatContext* input,
                        int inputVideoStream) {
  ff::installLogBridge();
  Impl& d = *d_;
  d.path = path;
  require(s.width > 0 && s.height > 0 && s.width % 2 == 0 && s.height % 2 == 0, ErrorCode::OutputError,
          "output dimensions must be positive and even");
  require(s.timeBase.valid(), ErrorCode::OutputError, "invalid encoder time base");

  int ret = avformat_alloc_output_context2(&d.oc, nullptr, nullptr, path.c_str());
  if (ret < 0 || !d.oc)
    fail(ErrorCode::OutputError, "cannot determine output format for '" + path + "' (use .mp4/.mkv/.mov): " +
                                     ff::errStr(ret));

  const AVCodec* codec = pickEncoder(s.codec);
  if (!codec) fail(ErrorCode::OutputError, "no usable video encoder found");
  d.stats.codecUsed = codec->name;
  d.enc.reset(avcodec_alloc_context3(codec));
  AVCodecContext* e = d.enc.get();
  e->width = s.width;
  e->height = s.height;
  e->pix_fmt = av_get_pix_fmt(s.pixFmt.c_str());
  require(e->pix_fmt == AV_PIX_FMT_YUV420P, ErrorCode::Unsupported,
          "only yuv420p output is currently supported (got '" + s.pixFmt + "')");
  d.srcTb = ff::toAV(s.timeBase);
  e->time_base = d.srcTb;
  if (s.frameRate.valid()) e->framerate = ff::toAV(s.frameRate);
  e->sample_aspect_ratio = ff::toAV(s.sampleAspect);
  ColorInfo col = s.color;
  inferColor(col, s.sourceHeight);
  e->color_primaries = static_cast<AVColorPrimaries>(col.primaries);
  e->color_trc = static_cast<AVColorTransferCharacteristic>(col.transfer);
  e->colorspace = static_cast<AVColorSpace>(col.space);
  e->color_range = static_cast<AVColorRange>(col.range);
  e->chroma_sample_location = static_cast<AVChromaLocation>(col.chromaLocation);
  e->thread_count = s.threads > 0 ? s.threads : std::clamp(sys::hardwareThreads() / 2, 2, 8);
  if (d.oc->oformat->flags & AVFMT_GLOBALHEADER) e->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

  AVDictionary* opts = nullptr;
  const std::string cname = codec->name;
  if (cname == "libx264" || cname == "libx265") {
    av_dict_set(&opts, "preset", s.preset.c_str(), 0);
    av_dict_set_int(&opts, "crf", s.crf, 0);
    if (cname == "libx265") av_dict_set(&opts, "x265-params", "log-level=error", 0);
  } else if (cname == "mpeg4") {
    e->flags |= AV_CODEC_FLAG_QSCALE;
    e->global_quality = FF_QP2LAMBDA * std::clamp(s.crf / 6, 1, 31);
  }
  ret = avcodec_open2(e, codec, &opts);
  av_dict_free(&opts);
  ff::check(ret, ErrorCode::OutputError, std::string("cannot open encoder ") + codec->name);

  d.vst = avformat_new_stream(d.oc, nullptr);
  if (!d.vst) fail(ErrorCode::OutputError, "cannot create output video stream");
  ff::check(avcodec_parameters_from_context(d.vst->codecpar, e), ErrorCode::OutputError, "stream params");
  d.vst->time_base = e->time_base;
  d.vst->avg_frame_rate = e->framerate;
  d.vst->sample_aspect_ratio = e->sample_aspect_ratio;
  if (s.rotationDeg != 0) {
    AVPacketSideData* sd = av_packet_side_data_new(&d.vst->codecpar->coded_side_data,
                                                   &d.vst->codecpar->nb_coded_side_data,
                                                   AV_PKT_DATA_DISPLAYMATRIX, sizeof(int32_t) * 9, 0);
    if (sd) av_display_rotation_set(reinterpret_cast<int32_t*>(sd->data), -static_cast<double>(s.rotationDeg));
  }

  // ---- stream-copy mapping for audio / subtitles ----
  if (input) {
    d.inToOut.assign(input->nb_streams, -1);
    d.inTimeBase.resize(input->nb_streams);
    if (input->streams[inputVideoStream]->metadata)
      av_dict_copy(&d.vst->metadata, input->streams[inputVideoStream]->metadata, 0);
    av_dict_set(&d.vst->metadata, "handler_name", "VideoUpscale", 0);
    for (unsigned i = 0; i < input->nb_streams; ++i) {
      const AVStream* ist = input->streams[i];
      const AVMediaType t = ist->codecpar->codec_type;
      const bool want = (t == AVMEDIA_TYPE_AUDIO && s.copyAudio) || (t == AVMEDIA_TYPE_SUBTITLE && s.copySubtitles);
      if (!want || static_cast<int>(i) == inputVideoStream) continue;
      const int q = avformat_query_codec(d.oc->oformat, ist->codecpar->codec_id, FF_COMPLIANCE_NORMAL);
      if (q == 0) {
        log::warn("input stream #{} ({} {}) cannot be stored in the '{}' container and will be dropped; "
                  "use a .mkv output to keep it",
                  i, av_get_media_type_string(t), avcodec_get_name(ist->codecpar->codec_id), d.oc->oformat->name);
        d.stats.droppedStreams.push_back(static_cast<int>(i));
        continue;
      }
      AVStream* ost = avformat_new_stream(d.oc, nullptr);
      if (!ost) fail(ErrorCode::OutputError, "cannot create output stream");
      ff::check(avcodec_parameters_copy(ost->codecpar, ist->codecpar), ErrorCode::OutputError, "copy params");
      ost->codecpar->codec_tag = 0;
      ost->time_base = ist->time_base;
      ost->disposition = ist->disposition;
      av_dict_copy(&ost->metadata, ist->metadata, 0);
      d.inToOut[i] = ost->index;
      d.inTimeBase[i] = ist->time_base;
      d.stats.copiedStreams.push_back(static_cast<int>(i));
    }
    if (input->metadata) av_dict_copy(&d.oc->metadata, input->metadata, 0);
  }
  if (!s.comment.empty()) av_dict_set(&d.oc->metadata, "comment", s.comment.c_str(), 0);
  av_dict_set(&d.oc->metadata, "encoder", nullptr, 0);

  if (!(d.oc->oformat->flags & AVFMT_NOFILE)) {
    ret = avio_open(&d.oc->pb, path.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) fail(ErrorCode::OutputError, "cannot open output file '" + path + "': " + ff::errStr(ret));
  }
  ret = avformat_write_header(d.oc, nullptr);
  ff::check(ret, ErrorCode::OutputError, "cannot write output header");
  d.headerWritten = true;

  d.frame = ff::makeFrame();
  d.frame->format = e->pix_fmt;
  d.frame->width = e->width;
  d.frame->height = e->height;
  ff::check(av_frame_get_buffer(d.frame.get(), 0), ErrorCode::OutputError, "frame buffer");
  d.pkt = ff::makePacket();
}

void VideoEncoder::write(const VideoFrame& f) {
  Impl& d = *d_;
  require(d.enc && !d.finished, ErrorCode::Generic, "encoder not open");
  require(f.width() == d.enc->width && f.height() == d.enc->height, ErrorCode::Generic,
          "frame size does not match encoder");
  ff::check(av_frame_make_writable(d.frame.get()), ErrorCode::OutputError, "frame writable");
  AVFrame* af = d.frame.get();
  const int cw = af->width / 2, ch = af->height / 2;
  f.y.copyTo(cv::Mat(af->height, af->width, CV_8UC1, af->data[0], af->linesize[0]));
  f.u(cv::Rect(0, 0, cw, ch)).copyTo(cv::Mat(ch, cw, CV_8UC1, af->data[1], af->linesize[1]));
  f.v(cv::Rect(0, 0, cw, ch)).copyTo(cv::Mat(ch, cw, CV_8UC1, af->data[2], af->linesize[2]));
  af->pts = f.pts;
  af->duration = f.duration;
  af->pict_type = AV_PICTURE_TYPE_NONE;
  int ret = avcodec_send_frame(d.enc.get(), af);
  ff::check(ret, ErrorCode::OutputError, "video encode (send) failed");
  d.drain();
  ++d.stats.framesWritten;
}

void VideoEncoder::copyPacket(AVPacket* pkt, int inputStreamIndex) {
  Impl& d = *d_;
  if (d.finished || inputStreamIndex < 0 || inputStreamIndex >= static_cast<int>(d.inToOut.size())) return;
  const int out = d.inToOut[inputStreamIndex];
  if (out < 0) return;
  AVStream* ost = d.oc->streams[out];
  av_packet_rescale_ts(pkt, d.inTimeBase[inputStreamIndex], ost->time_base);
  pkt->stream_index = out;
  pkt->pos = -1;
  d.writeMuxed(pkt);
  ++d.stats.copiedPackets;
}

void VideoEncoder::finish() {
  Impl& d = *d_;
  if (!d.enc || d.finished) return;
  ff::check(avcodec_send_frame(d.enc.get(), nullptr), ErrorCode::OutputError, "encoder flush");
  d.drain();
  {
    std::lock_guard lk(d.muxMutex);
    ff::check(av_write_trailer(d.oc), ErrorCode::OutputError, "cannot write trailer");
    d.finished = true;
  }
}

}  // namespace vu
