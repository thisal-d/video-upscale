# Third-Party Dependency Licenses

VideoUpscale links against the following open-source libraries (pinned in `scripts/fetch_deps.ps1`):

- **FFmpeg (`n8.1`, BtbN win64-gpl-shared)**: Built with `--enable-gpl --enable-version3` (includes `libx264` / `libx265`). Distributed binaries that include GPL components are subject to GPLv3 terms.
- **OpenCV (`4.14.0`)**: Apache 2.0 License.
- **nlohmann/json (`v3.12.0`)**: MIT License.
- **GoogleTest (`v1.18.0`)**: BSD 3-Clause License (test builds only).
- **Google Benchmark (`v1.10.0`)**: Apache 2.0 License (benchmark builds only).
