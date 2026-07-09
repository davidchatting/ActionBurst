# ActionBurst

A p5.js WEBGL sketch that aligns a burst photo sequence — using [opencv-featurematch-js](https://github.com/davidchatting/opencv-featurematch-js) to feature-match and RANSAC-align each frame against the nearest one before it — and plays it back with a fly-through camera, holding on each photo in turn before rewinding to the start.

**Live demo:** https://davidchatting.github.io/ActionBurst/

## How it works

- Photos are listed in `p5js/index.html`'s `#media` div. Each is ordered by its EXIF capture timestamp.
- On load, each new photo is aligned against the best-matching earlier one (nearest in time first), producing a 4x4 transform per photo.
- Playback fades through the photos in capture order, camera framing each one in turn, then rewinds directly back to the start.
- Once aligned, `#media`'s content can be copied out of the page (`#media-html`, gzip+base64-encoded) and baked back into `index.html` as a single `data-key` attribute on `#media` — this skips segmentation/alignment entirely on future loads.

## Editing

`p5js/` is also mirrored as a [p5.js Web Editor sketch](https://editor.p5js.org/davidchatting/sketches/91-r3FHTy) (see `package.json`'s `homepage`). Pushing changes there gets pulled back into this repo automatically by `.github/workflows/sync-p5js.yml`.

## License

MIT
