#include "io/decoder.hpp"

#include <algorithm>
#include <cmath>

#include "core/log.hpp"
#include "io/ffmpeg_utils.hpp"

namespace vu {

namespace {

int rotationOf(const AVStream* st) {
  const AVPacketSideData* sd = av_packet_side_data_get(st->codecpar->coded_side_data,
                                                       st->codecpar->nb_coded_side_data,
                                                       AV_PKT_DATA_DISPLAYMATRIX);
  if (!sd || sd->size < 9 * 4) return 0;
  double r = av_display_rotation_get(reinterpret_cast<const int32_t*>(sd->data));
  if (std::isnan(r)) return 0;
  int deg = static_cast<int>(std::lround(-r)) % 360;
  return deg < 0 ? deg + 360 : deg;
}

void fillInfo(VideoInfo& vi, AVFormatContext* fmt, int vidx, const AVCodecContext* dec) {
  const AVStream* st = fmt->streams[vidx];
  const AVCodecParameters* par = st->codecpar;
  vi.container = fmt->iformat ? fmt->iformat->name : "";
  vi.codec = avcodec_get_name(par->codec_id);
  const AVPixelFormat pf = dec ? dec->pix_fmt : static_cast<AVPixelFormat>(par->format);
  const char* pfn = av_get_pix_fmt_name(pf);
  vi.pixelFormat = pfn ? pfn : "unknown";
  if (const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(pf)) vi.bitDepth = desc->comp[0].depth;
  vi.width = par->width;
  vi.height = par->height;
  vi.timeBase = ff::toRational(st->time_base);
  vi.avgFrameRate = ff::toRational(st->avg_frame_rate);
  vi.realFrameRate = ff::toRational(st->r_frame_rate);
  if (!vi.avgFrameRate.valid()) vi.avgFrameRate = vi.realFrameRate;
  if (!vi.realFrameRate.valid()) vi.realFrameRate = vi.avgFrameRate;
  AVRational sar = av_guess_sample_aspect_ratio(fmt, const_cast<AVStream*>(st), nullptr);
  vi.sampleAspect = sar.num > 0 ? ff::toRational(sar) : Rational{1, 1};
  vi.startPts = st->start_time != AV_NOPTS_VALUE ? st->start_time : 0;
  if (st->duration != AV_NOPTS_VALUE && st->duration > 0)
    vi.durationSec = st->duration * av_q2d(st->time_base);
  else if (fmt->duration != AV_NOPTS_VALUE)
    vi.durationSec = fmt->duration / static_cast<double>(AV_TIME_BASE);
  vi.frameCountEstimate = st->nb_frames;
  if (vi.frameCountEstimate <= 0 && vi.durationSec > 0 && vi.avgFrameRate.valid())
    vi.frameCountEstimate = static_cast<int64_t>(std::llround(vi.durationSec * vi.avgFrameRate.toDouble()));
  vi.bitRate = par->bit_rate ? par->bit_rate : fmt->bit_rate;
  if (vi.avgFrameRate.valid() && vi.realFrameRate.valid()) {
    const double a = vi.avgFrameRate.toDouble(), r = vi.realFrameRate.toDouble();
    vi.likelyVfr = std::abs(a - r) / std::max(a, r) > 0.01;
  }
  vi.rotationDeg = rotationOf(st);
  vi.color.primaries = par->color_primaries;
  vi.color.transfer = par->color_trc;
  vi.color.space = par->color_space;
  vi.color.range = par->color_range;
  vi.color.chromaLocation = par->chroma_location;
  vi.videoStreamIndex = vidx;
  vi.audio.clear();
  vi.subtitleStreams = 0;
  for (unsigned i = 0; i < fmt->nb_streams; ++i) {
    const AVCodecParameters* p = fmt->streams[i]->codecpar;
    if (p->codec_type == AVMEDIA_TYPE_AUDIO) {
      AudioStreamInfo a;
      a.index = static_cast<int>(i);
      a.codec = avcodec_get_name(p->codec_id);
      a.sampleRate = p->sample_rate;
      a.channels = p->ch_layout.nb_channels;
      a.bitRate = p->bit_rate;
      vi.audio.push_back(a);
    } else if (p->codec_type == AVMEDIA_TYPE_SUBTITLE) {
      ++vi.subtitleStreams;
    }
  }
}

ff::FormatInputPtr openInput(const std::string& path) {
  AVFormatContext* raw = nullptr;
  int ret = avformat_open_input(&raw, path.c_str(), nullptr, nullptr);
  if (ret < 0) fail(ErrorCode::InputError, "cannot open input '" + path + "': " + ff::errStr(ret));
  ff::FormatInputPtr fmt(raw);
  ret = avformat_find_stream_info(fmt.get(), nullptr);
  if (ret < 0) fail(ErrorCode::InputError, "cannot read stream info of '" + path + "': " + ff::errStr(ret));
  return fmt;
}

int findVideoStream(AVFormatContext* fmt, const std::string& path) {
  int idx = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (idx >= 0 && (fmt->streams[idx]->disposition & AV_DISPOSITION_ATTACHED_PIC)) {
    idx = -1;
    for (unsigned i = 0; i < fmt->nb_streams; ++i) {
      const AVStream* s = fmt->streams[i];
      if (s->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && !(s->disposition & AV_DISPOSITION_ATTACHED_PIC)) {
        idx = static_cast<int>(i);
        break;
      }
    }
  }
  if (idx < 0) fail(ErrorCode::InputError, "no video stream found in '" + path + "'");
  return idx;
}

}  // namespace

struct VideoDecoder::Impl {
  ff::FormatInputPtr fmt;
  ff::CodecCtxPtr dec;
  ff::AVFramePtr frame;
  ff::AVPacketPtr pkt;
  ff::SwsPtr sws;
  int swsSrcFmt = -1, swsSrcW = 0, swsSrcH = 0;
  VideoInfo info;
  PacketCallback cb;
  bool demuxEof = false;
  bool decoderEof = false;
  int64_t decoded = 0;
  int64_t repairs = 0;
  int64_t lastPts = AV_NOPTS_VALUE;
  int64_t lastDur = 0;
  int64_t nominalDur = 1;  // one frame in stream time base units
  int64_t decodeErrors = 0;

  void convert(const AVFrame* f, VideoFrame& out) {
    require(out.width() == info.width && out.height() == info.height, ErrorCode::Generic,
            "output frame geometry mismatch");
    if (f->width != info.width || f->height != info.height)
      fail(ErrorCode::InputError, "resolution change mid-stream is not supported (" + std::to_string(f->width) +
                                      "x" + std::to_string(f->height) + ")");
    const auto pf = static_cast<AVPixelFormat>(f->format);
    if (pf == AV_PIX_FMT_YUV420P || pf == AV_PIX_FMT_YUVJ420P) {
      const int cw = (f->width + 1) / 2, ch = (f->height + 1) / 2;
      cv::Mat(f->height, f->width, CV_8UC1, f->data[0], f->linesize[0]).copyTo(out.y);
      cv::Mat(ch, cw, CV_8UC1, f->data[1], f->linesize[1]).copyTo(out.u);
      cv::Mat(ch, cw, CV_8UC1, f->data[2], f->linesize[2]).copyTo(out.v);
      return;
    }
    if (!sws || swsSrcFmt != f->format || swsSrcW != f->width || swsSrcH != f->height) {
      sws.reset(sws_getContext(f->width, f->height, pf, f->width, f->height, AV_PIX_FMT_YUV420P,
                               SWS_BICUBIC | SWS_ACCURATE_RND | SWS_FULL_CHR_H_INT, nullptr, nullptr, nullptr));
      if (!sws) fail(ErrorCode::InputError, std::string("unsupported pixel format: ") + av_get_pix_fmt_name(pf));
      swsSrcFmt = f->format;
      swsSrcW = f->width;
      swsSrcH = f->height;
      log::info("converting source pixel format {} -> yuv420p (8-bit processing)", av_get_pix_fmt_name(pf));
    }
    uint8_t* dst[4] = {out.y.data, out.u.data, out.v.data, nullptr};
    int dstStride[4] = {static_cast<int>(out.y.step), static_cast<int>(out.u.step), static_cast<int>(out.v.step), 0};
    sws_scale(sws.get(), f->data, f->linesize, 0, f->height, dst, dstStride);
  }

  void stampTimes(const AVFrame* f, VideoFrame& out) {
    int64_t ts = f->best_effort_timestamp;
    if (ts == AV_NOPTS_VALUE) ts = f->pts;
    if (ts == AV_NOPTS_VALUE) {
      ts = (lastPts == AV_NOPTS_VALUE) ? 0 : lastPts + (lastDur > 0 ? lastDur : nominalDur);
      ++repairs;
    } else if (lastPts != AV_NOPTS_VALUE && ts <= lastPts) {
      // Non-monotonic presentation timestamps would make the muxer fail: nudge forward.
      ts = lastPts + 1;
      ++repairs;
    }
    int64_t dur = f->duration > 0 ? f->duration : nominalDur;
    out.pts = ts;
    out.duration = dur;
    out.index = decoded;
    lastPts = ts;
    lastDur = dur;
  }
};

VideoDecoder::VideoDecoder() : d_(std::make_unique<Impl>()) {}
VideoDecoder::~VideoDecoder() = default;

void VideoDecoder::open(const std::string& path, const DecoderOptions& opt) {
  ff::installLogBridge();
  d_->fmt = openInput(path);
  const int vidx = findVideoStream(d_->fmt.get(), path);
  AVStream* st = d_->fmt->streams[vidx];
  const AVCodec* codec = avcodec_find_decoder(st->codecpar->codec_id);
  if (!codec)
    fail(ErrorCode::InputError, std::string("no decoder for codec ") + avcodec_get_name(st->codecpar->codec_id));
  d_->dec.reset(avcodec_alloc_context3(codec));
  ff::check(avcodec_parameters_to_context(d_->dec.get(), st->codecpar), ErrorCode::InputError, "decoder params");
  d_->dec->pkt_timebase = st->time_base;
  // Bounded decoder threading: the processing stage owns most cores.
  d_->dec->thread_count = opt.threads > 0 ? opt.threads : 4;
  d_->dec->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
  ff::check(avcodec_open2(d_->dec.get(), codec, nullptr), ErrorCode::InputError, "cannot open video decoder");
  d_->frame = ff::makeFrame();
  d_->pkt = ff::makePacket();
  d_->info = VideoInfo{};
  d_->info.path = path;
  fillInfo(d_->info, d_->fmt.get(), vidx, d_->dec.get());
  require(d_->info.width > 0 && d_->info.height > 0, ErrorCode::InputError, "invalid video dimensions");
  // Nominal frame duration in stream time-base units (used only to repair missing timestamps).
  Rational fr = d_->info.avgFrameRate.valid() ? d_->info.avgFrameRate : Rational{25, 1};
  d_->nominalDur = std::max<int64_t>(1, av_rescale_q(1, av_inv_q(ff::toAV(fr)), st->time_base));
  for (unsigned i = 0; i < d_->fmt->nb_streams; ++i)
    if (static_cast<int>(i) != vidx) d_->fmt->streams[i]->discard = AVDISCARD_DEFAULT;
}

const VideoInfo& VideoDecoder::info() const { return d_->info; }
void VideoDecoder::setPacketCallback(PacketCallback cb) { d_->cb = std::move(cb); }
const AVFormatContext* VideoDecoder::formatContext() const { return d_->fmt.get(); }
int64_t VideoDecoder::framesDecoded() const { return d_->decoded; }
int64_t VideoDecoder::timestampRepairs() const { return d_->repairs; }

bool VideoDecoder::read(VideoFrame& out) {
  require(d_->dec != nullptr, ErrorCode::Generic, "decoder not opened");
  Impl& d = *d_;
  const int vidx = d.info.videoStreamIndex;
  while (true) {
    if (d.decoderEof) return false;
    int ret = avcodec_receive_frame(d.dec.get(), d.frame.get());
    if (ret >= 0) {
      d.convert(d.frame.get(), out);
      d.stampTimes(d.frame.get(), out);
      av_frame_unref(d.frame.get());
      ++d.decoded;
      return true;
    }
    if (ret == AVERROR_EOF) {
      d.decoderEof = true;
      if (d.decodeErrors) log::warn("{} video packets failed to decode (corrupt input?)", d.decodeErrors);
      return false;
    }
    if (ret != AVERROR(EAGAIN)) ff::check(ret, ErrorCode::InputError, "video decode failed");

    // Decoder needs more input.
    if (d.demuxEof) {
      ff::check(avcodec_send_packet(d.dec.get(), nullptr), ErrorCode::InputError, "decoder flush");
      continue;
    }
    ret = av_read_frame(d.fmt.get(), d.pkt.get());
    if (ret == AVERROR_EOF || ret == AVERROR(EIO) && avio_feof(d.fmt->pb)) {
      d.demuxEof = true;
      continue;
    }
    ff::check(ret, ErrorCode::InputError, "demux failed");
    if (d.pkt->stream_index == vidx) {
      ret = avcodec_send_packet(d.dec.get(), d.pkt.get());
      if (ret < 0 && ret != AVERROR(EAGAIN)) {
        ++d.decodeErrors;
        log::debug("decode error on packet pts={}: {}", d.pkt->pts, ff::errStr(ret));
      }
    } else if (d.cb) {
      d.cb(d.pkt.get(), d.pkt->stream_index);
    }
    av_packet_unref(d.pkt.get());
  }
}

VideoInfo probeVideo(const std::string& path) {
  ff::installLogBridge();
  auto fmt = openInput(path);
  const int vidx = findVideoStream(fmt.get(), path);
  VideoInfo vi;
  vi.path = path;
  fillInfo(vi, fmt.get(), vidx, nullptr);
  return vi;
}

}  // namespace vu
