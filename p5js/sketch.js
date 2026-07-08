// identityMatrix, applyTransform4x4, multiplyMatrix4x4, invertMatrix4x4 and
// to2dAffine live in shimage.js; stripShear lives in opencv-featurematch-js.js
const imageTransforms = [];
let mediaBoundingBox = null;

// Once a candidate match clears this many RANSAC inliers, stop searching further
// (temporally-nearer) candidates — an early exit to avoid an O(n^2) alignment
// search across the whole sequence when the nearest neighbour is already a
// confident match, which is true for the large majority of burst-sequence pairs.
const EARLY_EXIT_INLIER_THRESHOLD = 50;

let maskSegmentation = null;

// Real-time playback of the capture sequence, driven by each image's EXIF timestamp.
const PLAYBACK_START_PAUSE_MS = 3000;
const PLAYBACK_END_PAUSE_MS = 3000;
const PLAYBACK_SPEED = 0.5; // 1 = real-time (matches original capture pace), 0.5 = half speed
let playbackSchedule = [];
let playbackStartMillis = 0;

// The image currently shown in full colour — whichever most recently started
// its own hold. It persists (rather than reverting) once its hold/fade ends,
// until the next image's hold begins and takes over. Every other image still
// shows, desaturated, in the background (see draw()).
let currentDisplayIndex = -1;

// Space bar pauses/resumes playback. Pausing just freezes the clock that
// drives everything else (getPlaybackElapsedMs) rather than touching
// playbackStartMillis directly, so resuming continues exactly where it left
// off instead of jumping.
let isPaused = false;
let pauseBeganAtMillis = 0;
let totalPausedMs = 0;

function getEffectiveMillis() {
  const now = isPaused ? pauseBeganAtMillis : millis();
  return now - totalPausedMs;
}

// Used only for images with no recoverable EXIF timestamp, to keep the
// playback schedule well-formed (monotonically increasing) when falling back
// to alphabetical-filename ordering.
const FALLBACK_FRAME_SPACING_MS = 500;

// Once the timeline reaches a given image's own scheduled moment, it holds
// at full opacity for this long afterward (real wall-clock ms, NOT scaled by
// PLAYBACK_SPEED) — never before its own time, only after. Every image still
// shows faintly at all times via the constant low alpha in draw().
const HOLD_MS = 500;

// Alpha jumps instantly to full opacity when a photo's hold begins, then
// ramps smoothly back down to LOW_ALPHA over this much of the end of the
// hold window (real ms, same convention as HOLD_MS). Clamped to the hold
// window's own length if HOLD_MS is shorter.
const FADE_MS = 150;

// The constant background level every image sits at outside its own hold window.
const LOW_ALPHA = 0.1;

// 3D camera fly-through: one keyframe per aligned image, framing that image
// alone, timed to the exact same clock as playbackSchedule above — the camera
// is exactly centred on an image at the moment it becomes the current frame,
// then travels to be exactly framed on the next image by its due time.
let cameraKeyframes = [];

/**
 * Creates the foreground segmenter and waits until it's ready.
 * Returns a Promise that resolves when the segmenter is ready.
 */
async function createForegroundSegmenter() {
  maskSegmentation = new SelfieSegmentation({
    locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation@latest/${file}`;
    }
  });
  var options = {
    selfieMode: true,
    modelSelection: 0, // general
    effect: 'mask',
  };
  maskSegmentation.setOptions(options);

  // Wait for the WASM to load (simulate "blocking" until ready)
  await maskSegmentation.initialize();
}

// In setup(), use async/await to block until ready
async function setup() {
  // Use WEBGL so texture()/vertex(u,v) in drawImageWithHomography works
  canvas = createCanvas(2000, 800, WEBGL);
  canvas.parent('canvas-target'); // explicit target, rather than p5's default append-to-body
  frameRate(10);
  canvas.drop(onFileDropped);

  // Block until segmenter is ready
  await createForegroundSegmenter();

  // opencv.js's <script onload> fires once its JS wrapper has loaded, not
  // once its WASM runtime has actually finished initializing - wait for both
  // it and shimage.js before anything below can call into either.
  await featurematchReady();

  // ensure texture UVs use normalized coordinates
  textureMode(NORMAL);

  processAnyAttachedMedia();
}

// Mirrors warnings onto the page itself (#on-screen-console), not just the
// browser devtools console — this sketch is often run/projected without
// devtools open, so alignment failures would otherwise go unseen.
function logWarning(...args) {
  console.warn(...args);

  const consoleElement = select('#on-screen-console')?.elt;
  if (!consoleElement) return;

  const line = document.createElement('div');
  line.textContent = args.map(a => (typeof a === 'string' ? a : JSON.stringify(a))).join(' ');
  consoleElement.appendChild(line);
}

function onFileDropped(file) {
  const id = file.name;
  console.log("Dropped file: " + file.name);
  const div = upsertMedia(id);

  const originalImg = createImg(file.data, '', () => {
    originalImg.parent(div);
    originalImg.addClass('original');
    setImageTransform(originalImg.elt, identityMatrix);

    processImage(originalImg.elt, div);
  });
}

function setImageTransform(element, transform) {
  if (element && Array.isArray(transform)) {
    element.setAttribute('data-transform', JSON.stringify(transform));
  }
}

function getImageTransformFromElement(element, traverse = false) {
  let result = null;

  if (element) {
    const b = traverse ? (getImageTransformFromElement(element.parentElement, false) || identityMatrix) : identityMatrix;
    try {
      result = JSON.parse(element.getAttribute('data-transform'));
    }
    catch (e) {
    }
    if (result) result = multiplyMatrix4x4(b, result);
  }

  return result;
}

function generateLowResImage(imgElement, onloaded = () => {}) {
  let lowresImg = null;

  const lowresMaxPixels = 1024 * 768;
  if (imgElement.width * imgElement.height > lowresMaxPixels) {
    const s = Math.sqrt(lowresMaxPixels / (imgElement.width * imgElement.height));

    const targetW = Math.round(imgElement.width * s);
    const targetH = Math.round(imgElement.height * s);

    const canvas = document.createElement('canvas');
    canvas.width = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(imgElement, 0, 0, targetW, targetH);

    const dataUrl = canvas.toDataURL("image/jpeg", 1.0);
    canvas.width = 0;
    canvas.height = 0;

    lowresImg = createImg(dataUrl, '');
    lowresImg.elt.onload = onloaded;

    // Attach a 4x4 scaling transform (column-major; diagonal-only here so
    // identical either way)
    const invS = 1 / s;
    const scaleTransform = [
      invS, 0, 0, 0,
      0, invS, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1
    ];
    setImageTransform(lowresImg.elt, scaleTransform);
  } else {
    lowresImg = new p5.Element(imgElement);
    setTimeout(onloaded, 0);

    // Attach identity transform (no scaling)
    setImageTransform(lowresImg.elt, identityMatrix);
  }

  return lowresImg;
}

// Helper: Promise version of generateLowResImage
function generateLowResImageAsync(imgElement) {
  return new Promise(resolve => {
    const lowresImg = generateLowResImage(imgElement, () => resolve(lowresImg));
  });
}

// Expects a ready MediaPipe SelfieSegmentation instance in the global `maskSegmentation`.
function generateMask(imgElement, onloaded = () => {}) {
  let maskImg = createImg('', '');

  maskSegmentation.onResults(async (results) => {
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = results.segmentationMask.width;
    maskCanvas.height = results.segmentationMask.height;
    const ctx = maskCanvas.getContext('2d');

    // flip horizontally
    ctx.translate(maskCanvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(results.segmentationMask, 0, 0);

    // convert red-channel mask to greyscale (copy R to G and B)
    const imageData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];     // red channel holds the mask value
      data[i]     = r;       // R (keep)
      data[i + 1] = r;       // G (copy from R)
      data[i + 2] = r;       // B (copy from R)
    }
    ctx.putImageData(imageData, 0, 0);
    setImageTransform(maskImg.elt, getImageTransformFromElement(imgElement));

    maskImg.elt.onload = onloaded;
    maskImg.elt.src = maskCanvas.toDataURL();
  });
  maskSegmentation.send({ image: imgElement });

  return (maskImg);
}

// Helper: Promise version of generateMask
function generateMaskAsync(imgElement) {
  return new Promise(resolve => {
    const maskImg = generateMask(imgElement, () => resolve(maskImg));
  });
}

/**
 * Creates a new image element with the mask applied.
 * Pixels where the mask is dark (black) become transparent.
 * @param {HTMLImageElement|p5.Element} colorImg - the colour image
 * @param {HTMLImageElement|p5.Element} maskImg - the greyscale mask (white = keep, black = transparent)
 * @returns {p5.Element} - a new p5 img element containing the masked image
 */
function applyMaskToImage(colorImg, maskImg, invert = false, onloaded = () => {}) {
  let resultImg = createImg('', '');

  const w = colorImg.naturalWidth || colorImg.width;
  const h = colorImg.naturalHeight || colorImg.height;

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  ctx.drawImage(colorImg, 0, 0, w, h);

  const colorData = ctx.getImageData(0, 0, w, h);
  const cPixels = colorData.data;

  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(maskImg, 0, 0, w, h);
  const maskData = ctx.getImageData(0, 0, w, h);
  const mPixels = maskData.data;

  for (let i = 0; i < cPixels.length; i += 4) {
    const maskVal = invert ? 255 - mPixels[i] : mPixels[i];
    cPixels[i] = maskVal > 0 ? cPixels[i] : random(255);
    cPixels[i + 1] = maskVal > 0 ? cPixels[i + 1] : random(255);
    cPixels[i + 2] = maskVal > 0 ? cPixels[i + 2] : random(255);
    cPixels[i + 3] = maskVal;
  }

  ctx.putImageData(colorData, 0, 0);
  setImageTransform(resultImg.elt, getImageTransformFromElement(colorImg));

  resultImg.elt.onload = onloaded;
  resultImg.elt.src = canvas.toDataURL();

  return resultImg;
}

// Helper: Promise version of applyMaskToImage
function applyMaskToImageAsync(colorImg, maskImg, invert) {
  return new Promise(resolve => {
    const resultImg = applyMaskToImage(colorImg, maskImg, invert, () => resolve(resultImg));
  });
}

// processHomography is defined further down. getTextureFromElement and
// drawProjectedImage come from shimage.js (already loaded).

// Cache of desaturated (greyscale) copies of images, keyed by the original
// element — computed once and reused. tint() alone can only dim/tint a
// texture, not actually desaturate it, so background images get a genuinely
// greyscale copy drawn instead of the colour original. Uses a plain 2D
// canvas pixel loop rather than p5's built-in filter(GRAY) on a
// createGraphics() buffer — that threw a WebGL "useProgram" error and broke
// the whole canvas, since it shares/conflicts with the main sketch's own
// WEBGL context.
const greyscaleCache = new WeakMap();

function getGreyscaleElement(img) {
  if (greyscaleCache.has(img)) return greyscaleCache.get(img);

  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);

  const imageData = ctx.getImageData(0, 0, w, h);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    const grey = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    data[i] = data[i + 1] = data[i + 2] = grey;
  }
  ctx.putImageData(imageData, 0, 0);

  greyscaleCache.set(img, canvas);
  return canvas;
}

// drawProjectedImage now lives in shimage.js

function upsertMedia(id) {
  if (!id) return null;

  let container = select('#media');
  if (!container) return null;

  const found = container.elt.querySelector('#' + id);
  if (found) return select('#' + id); // use p5.select to return a p5.Element

  const d = createDiv('');
  d.id(id);
  d.parent(container);
  return d;
}

/**
 * Returns the bounding box (in screen coordinates) that contains all media elements,
 * with their transforms applied (using imageTransforms).
 * @returns {{left: number, top: number, right: number, bottom: number}|null}
 */
function getBoundingBox(selector, indices) {
  const mediaElement = select('#media')?.elt;
  if (!mediaElement) return null;

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

  for (const i of indices) {
    const transform = getImageTransformFromElement(mediaElement.children[i], true);
    if (!transform) continue;

    const image = mediaElement.children[i].querySelector(selector);
    if (!image) continue;

    const w = image.naturalWidth || image.width;
    const h = image.naturalHeight || image.height;

    // Corners in local image coordinates (top-left origin)
    const corners = [
      [0, 0],
      [w, 0],
      [w, h],
      [0, h]
    ];

    // Transform each corner and update bounds
    for (const [x, y] of corners) {
      const [tx, ty] = applyTransform4x4(x, y, transform);
      minX = Math.min(minX, tx);
      minY = Math.min(minY, ty);
      maxX = Math.max(maxX, tx);
      maxY = Math.max(maxY, ty);
    }
  }

  if (minX === Infinity) return null;

  return { left: minX, top: minY, width: maxX - minX, height: maxY - minY };
}

function draw() {
  const imageSelector = '.original';
  background(220);

  // Don't animate until every image has been through the segmentation/alignment
  // pipeline and the playback schedule is ready — the raw #media images (now
  // visible in the page) show progress until then.
  if (playbackSchedule.length === 0) return;

  const scheduledIndices = playbackSchedule.map(e => e.index);
  mediaBoundingBox = getBoundingBox(imageSelector, scheduledIndices);

  // The most recently active image fades smoothly up to full colour opacity
  // right as its own scheduled moment begins (never before), then back down
  // to LOW_ALPHA, and simply persists highlighted once its own fade ends
  // until the next image's hold takes over. Rewind never starts a new hold —
  // it's just the camera travelling back to the start.
  const elapsed = getPlaybackElapsedMs();
  const rewinding = isRewinding();
  const holdWindow = HOLD_MS * PLAYBACK_SPEED;
  const fadeWindow = Math.min(FADE_MS * PLAYBACK_SPEED, holdWindow);

  const alphas = playbackSchedule.map(entry => rewinding ? LOW_ALPHA : holdAlphaAt(elapsed, entry.offsetMs, holdWindow, fadeWindow));

  let highlightPos = 0;
  for (let p = 1; p < alphas.length; p++) {
    if (alphas[p] > alphas[highlightPos]) highlightPos = p;
  }
  const highlighted = alphas[highlightPos] > LOW_ALPHA;
  if (highlighted) currentDisplayIndex = playbackSchedule[highlightPos].index;

  updateDebugTimeDisplay(elapsed, highlighted ? currentDisplayIndex : -1);

  // The camera is always continuously interpolating between keyframes,
  // independent of which image (if any) is currently held at full opacity —
  // it keeps moving throughout every hold window rather than freezing on a
  // single keyframe, and only actually comes to rest during the real
  // end-of-sequence pause (getCameraPose() naturally settles on the last
  // keyframe there, since elapsed is pushed beyond every keyframe's time).
  const camPose = getCameraPose();
  if (camPose) {
    camera(
      camPose.eye[0], camPose.eye[1], camPose.eye[2],
      camPose.center[0], camPose.center[1], camPose.center[2],
      camPose.up[0], camPose.up[1], camPose.up[2]
    );
  }

  // Every other (unhighlighted) image still shows in the background, but
  // desaturated to greyscale at the constant low alpha, so the current
  // highlighted image (full colour) reads clearly as "the one in focus".
  // Depth writes are disabled — otherwise the nearer quad's depth value
  // would block farther ones from blending through underneath it.
  const mediaElement = select('#media')?.elt;
  if (mediaElement) {
    push();
      drawingContext.depthMask(false);

      for (let p = 0; p < playbackSchedule.length; p++) {
        const entry = playbackSchedule[p];
        if (entry.index === currentDisplayIndex) continue;
        const image = mediaElement.children[entry.index].querySelector(imageSelector);
        if (!image) continue;

        push();
          tint(255, 255 * LOW_ALPHA);
          const t = stripShear(getImageTransformFromElement(image, true));
          drawProjectedImage(getGreyscaleElement(image), 0, 0, t, -p);
        pop();
      }

      if (currentDisplayIndex >= 0) {
        const pos = playbackSchedule.findIndex(e => e.index === currentDisplayIndex);
        const image = mediaElement.children[currentDisplayIndex].querySelector(imageSelector);
        if (image) {
          push();
            tint(255, 255 * (pos >= 0 ? alphas[pos] : LOW_ALPHA));
            const t = stripShear(getImageTransformFromElement(image, true));
            drawProjectedImage(image, 0, 0, t, 0);
          pop();
        }
      }

      drawingContext.depthMask(true);
    pop();
  }
}

// applyMaskToImage(+Async) is defined above. isReasonableHomography lives in
// opencv-featurematch-js.js; applyTransform4x4 and the other 4x4 matrix
// helpers (multiply/invert/determinant/cofactor) live in shimage.js

function keyPressed() {
  if (key === 'x' || key === 'X') {
    exportAllMediaElements('.original');
  }

  if (key === ' ') {
    if (isPaused) {
      totalPausedMs += millis() - pauseBeganAtMillis;
      isPaused = false;
    } else {
      pauseBeganAtMillis = millis();
      isPaused = true;
    }
    return false; // prevent the browser's default page-scroll-on-space
  }
}

/**
 * Creates as many fully transparent PNG image elements as there are media elements,
 * each with the dimensions of mediaBoundingBox, draws the corresponding media image into it
 * using its transform relative to the bounding box, and appends them to #export.
 */
function exportAllMediaElements(selector) {
  const mediaElement = select('#media')?.elt;
  const exportElement = select('#export')?.elt;
  if (!mediaElement || !exportElement || !mediaBoundingBox) {
    logWarning('Missing #media, #export, or mediaBoundingBox');
    return;
  }

  // Clear previous exports
  exportElement.innerHTML = '';

  const w = Math.round(mediaBoundingBox.width);
  const h = Math.round(mediaBoundingBox.height);

  for (let i = 0; i < mediaElement.children.length; i++) {
    // Create a canvas for export
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');

    // Fill with fully transparent background
    ctx.clearRect(0, 0, w, h);

    // Draw the media image (e.g. .lowres) into the canvas, using its transform
    const img = mediaElement.children[i].querySelector(selector);
    const transform = getImageTransformFromElement(img, true);
    console.log(transform);
    if (img && transform) {
      const imgW = img.naturalWidth || img.width;
      const imgH = img.naturalHeight || img.height;

      // Extract 2D affine for canvas setTransform(a, b, c, d, e, f)
      const [a, b, c, d, e, f] = to2dAffine(transform);
      const tx = e - mediaBoundingBox.left;
      const ty = f - mediaBoundingBox.top;

      ctx.save();
      ctx.setTransform(a, b, c, d, tx, ty);
      ctx.drawImage(img, 0, 0, imgW, imgH);
      ctx.restore();
    }

    // Create an image element from the canvas
    const outImg = document.createElement('img');
    outImg.src = canvas.toDataURL('image/png');
    outImg.width = w;
    outImg.height = h;
    outImg.style.width = w + "px";
    outImg.style.height = h + "px";

    exportElement.appendChild(outImg);
  }
}

// Orders two images by their recovered capture sequence: EXIF timestamp when
// both have one, otherwise alphabetically by filename (images with a
// timestamp always sort before ones without).
function compareImagesForSequence(imgA, imgB) {
  const ta = getImageTimestampFromElement(imgA);
  const tb = getImageTimestampFromElement(imgB);
  if (ta && tb) return ta.getTime() - tb.getTime();
  if (ta) return -1;
  if (tb) return 1;
  return imgA.src.localeCompare(imgB.src);
}

/**
 * Aligns the newest image in a '.background'-tagged media collection against
 * the best-matching (by RANSAC inlier count) previously-aligned image, trying
 * candidates nearest-in-time first and stopping early once a confident match
 * is found. Writes the resulting 4x4 transform onto the new image's container
 * via setImageTransform. This search strategy (which candidate to try first,
 * when to stop) is a RugbySynth-specific policy, not part of the generic
 * imgproc.js library.
 */
function processHomography(id) {
  const selector = '.background';
  const mediaCollection = select('#media')?.elt.querySelectorAll(selector);
  if (!mediaCollection || mediaCollection.length === 0) return;

  const n = mediaCollection.length;

  if (n === 1) {
    setImageTransform(mediaCollection[0].parentElement, identityMatrix);
    return;
  }

  // The newest image is the last one
  const image_b = mediaCollection[n - 1];

  // Skip if already aligned
  if (getImageTransformFromElement(image_b.parentElement)) return;

  // Try all previously aligned images and pick the best match by inlier count
  let bestInliers = 0;
  let bestT0B = null;
  let bestMatchId = null;

  for (let i = n - 2; i >= 0; i--) {
    const image_a = mediaCollection[i];
    const t0A = getImageTransformFromElement(image_a.parentElement);

    // Skip images that haven't been aligned yet
    if (!t0A) continue;

    // alignImages(a, b) now maps a -> b (flipped from the old alignImagePair,
    // which mapped its second argument onto its first) - called here swapped,
    // as (image_b, image_a), so result.transform still maps b -> a, matching
    // the tAa/tBb composition below.
    const result = alignImages(image_b, image_a);
    const inliers = result.inlierMatches.length;

    if (result.valid && inliers > bestInliers) {
      const tAa = getImageTransformFromElement(image_a);
      const tBb = getImageTransformFromElement(image_b);
      const tBb_i = invertMatrix4x4(tBb);
      const tAB = multiplyMatrix4x4(multiplyMatrix4x4(tAa, result.transform), tBb_i);

      bestT0B = multiplyMatrix4x4(t0A, tAB);
      bestInliers = inliers;
      bestMatchId = image_a.parentElement.id;

      // Candidates are tried nearest-in-time first (i counts down from n-2),
      // so a confident match here is very likely the best one available —
      // stop searching rather than aligning against every earlier frame too.
      if (bestInliers >= EARLY_EXIT_INLIER_THRESHOLD) {
        break;
      }
    } else if (!result.valid) {
      logWarning('Rejecting homography with', image_a.parentElement.id, ':', result.reason);
    }
  }

  if (bestT0B) {
    setImageTransform(image_b.parentElement, bestT0B);
  } else {
    logWarning('No valid homography found for', image_b.parentElement.id);
  }
}

async function processAnyAttachedMedia() {
  // If #media itself carries a data-key (the compressed, base64-encoded
  // output revealMediaForCopying() produces), decode it and use it to
  // populate #media's children before anything else runs - this lets a
  // page ship just that one compact attribute instead of the full,
  // already-aligned div/img markup, and processImage()'s own
  // already-aligned check then skips re-running segmentation on it.
  const mediaElement = select('#media')?.elt;
  const key = mediaElement?.getAttribute('data-key');
  if (key) {
    mediaElement.innerHTML = await decompressFromBase64(key);
  }

  const originals = selectAll('#media .original');
  // Wait for all images to load
  await Promise.all(originals.map(i => {
    return new Promise(resolve => {
      if (i.elt.complete) resolve();
      else {
        i.elt.onload = resolve;
        i.elt.onerror = resolve;
      }
    });
  }));

  // Phase 1: recover every image's EXIF timestamp up front, before any
  // alignment work — the processing order below (and buildPlaybackSchedule
  // afterwards) is derived from these.
  for (const orig of originals) {
    setImageTransform(orig.elt, identityMatrix);
    const timestamp = await extractImageTimestamp(orig.elt);
    setImageTimestamp(orig.elt, timestamp);
  }

  // Phase 2: segment + align in chronological (or alphabetical-fallback)
  // order, so each new image is matched against the nearest one actually
  // preceding it in the recovered sequence.
  const ordered = [...originals].sort((a, b) => compareImagesForSequence(a.elt, b.elt));
  for (const orig of ordered) {
    await processImage(orig.elt, orig.parent());
    processHomography(orig.parent().id);
  }

  buildPlaybackSchedule();
  buildCameraKeyframes();

  await revealMediaForCopying();
}

// #media stays hidden (see style.css) — by this point each wrapping div's
// data-transform attribute holds the real alignment matrix (the .original
// img's own data-transform is always just the identity, see
// processAnyAttachedMedia's Phase 1 and processHomography). The
// .lowres/.mask/.foreground/.background img elements processImage() also
// leaves on each div are pipeline intermediates carrying base64 image data
// - stripped here, then the remaining markup (each div plus its .original
// img) is gzip-compressed and base64-encoded into #media-html, for copying
// out of the page as a compact string.
async function revealMediaForCopying() {
  const mediaElement = select('#media')?.elt;
  const outputElement = select('#media-html')?.elt;
  if (!outputElement) return;

  if (!mediaElement) {
    outputElement.textContent = 'Error: #media not found';
    return;
  }

  const clone = mediaElement.cloneNode(true);
  clone.querySelectorAll('img:not(.original)').forEach(img => img.remove());
  outputElement.textContent = await compressToBase64(clone.innerHTML);
}

function bytesToBase64(bytes) {
  let binary = '';
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary);
}

function base64ToBytes(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

// gzip-compresses text via the browser's built-in CompressionStream, then
// base64-encodes the compressed bytes into a single copyable string.
async function compressToBase64(text) {
  const cs = new CompressionStream('gzip');
  const writer = cs.writable.getWriter();
  writer.write(new TextEncoder().encode(text));
  writer.close();

  const chunks = [];
  const reader = cs.readable.getReader();
  for (let result = await reader.read(); !result.done; result = await reader.read()) {
    chunks.push(result.value);
  }

  const compressedBytes = new Uint8Array(chunks.reduce((sum, c) => sum + c.length, 0));
  let offset = 0;
  for (const chunk of chunks) {
    compressedBytes.set(chunk, offset);
    offset += chunk.length;
  }

  return bytesToBase64(compressedBytes);
}

// Inverse of compressToBase64(): base64-decodes then gunzips via
// DecompressionStream, back to the original text.
async function decompressFromBase64(base64) {
  const ds = new DecompressionStream('gzip');
  const writer = ds.writable.getWriter();
  writer.write(base64ToBytes(base64));
  writer.close();

  const chunks = [];
  const reader = ds.readable.getReader();
  for (let result = await reader.read(); !result.done; result = await reader.read()) {
    chunks.push(result.value);
  }

  const decompressedBytes = new Uint8Array(chunks.reduce((sum, c) => sum + c.length, 0));
  let offset = 0;
  for (const chunk of chunks) {
    decompressedBytes.set(chunk, offset);
    offset += chunk.length;
  }

  return new TextDecoder().decode(decompressedBytes);
}

// Orders images by their recovered capture sequence and records each one's
// offset (in ms) from the first frame, so draw()/the camera can play the
// sequence back at the same pace it was actually shot.
function buildPlaybackSchedule() {
  const mediaElement = select('#media')?.elt;
  if (!mediaElement) return;

  const entries = [];
  for (let i = 0; i < mediaElement.children.length; i++) {
    const image = mediaElement.children[i].querySelector('.original');
    if (image) entries.push({ index: i, div: mediaElement.children[i], image });
  }
  if (entries.length === 0) return;

  entries.sort((a, b) => compareImagesForSequence(a.image, b.image));

  // Stop the sequence at the first frame with no valid alignment transform,
  // rather than including a mispositioned/unaligned frame in playback.
  const aligned = [];
  for (const e of entries) {
    if (!getImageTransformFromElement(e.div)) break;
    aligned.push(e);
  }
  if (aligned.length === 0) return;

  const t0 = getImageTimestampFromElement(aligned[0].image)?.getTime();

  let lastOffset = -FALLBACK_FRAME_SPACING_MS;
  playbackSchedule = aligned.map(e => {
    const t = getImageTimestampFromElement(e.image)?.getTime();
    const offsetMs = (t !== undefined && t0 !== undefined) ? (t - t0) : (lastOffset + FALLBACK_FRAME_SPACING_MS);
    lastOffset = offsetMs;
    return { index: e.index, offsetMs };
  });
  playbackStartMillis = millis();
}

// Rewind travels back to the start this many times faster than forward
// playback — a quick single motion rather than a full-speed mirror of the
// forward pass.
const REWIND_SPEED_MULTIPLIER = 2;

// Shared phase arithmetic for getPlaybackElapsedMs() and isRewinding(), all
// using the pausable getEffectiveMillis() clock. Five phases:
//  1. Start-of-sequence pause: always PLAYBACK_START_PAUSE_MS of real
//     wall-clock time regardless of speed — a beat before the first photo's
//     hold begins, camera parked at the first keyframe, mirroring the pause
//     at the end.
//  2. Forward: 0 -> lastOffset, scaled by PLAYBACK_SPEED.
//  3. Extended forward: time keeps running normally (same PLAYBACK_SPEED
//     scaling, no special-casing) for a further HOLD_MS of real time past
//     lastOffset — otherwise the last photo's hold window would be cut off
//     the instant it's reached, while every other photo gets its full
//     HOLD_MS afterward.
//  4. End-of-sequence pause: always PLAYBACK_END_PAUSE_MS of real wall-clock
//     time regardless of speed.
//  5. Rewind: lastOffset -> 0, at REWIND_SPEED_MULTIPLIER x the forward speed.
function getPlaybackPhaseInfo() {
  const lastOffset = playbackSchedule[playbackSchedule.length - 1].offsetMs;
  const realTravelDuration = lastOffset / PLAYBACK_SPEED;
  const realExtendedTravelDuration = realTravelDuration + HOLD_MS;
  const rewindSpeed = PLAYBACK_SPEED * REWIND_SPEED_MULTIPLIER;
  const realRewindDuration = lastOffset / rewindSpeed;
  const realCycleLength = PLAYBACK_START_PAUSE_MS + realExtendedTravelDuration + PLAYBACK_END_PAUSE_MS + realRewindDuration;
  const realElapsed = (getEffectiveMillis() - playbackStartMillis) % realCycleLength;

  return { lastOffset, realExtendedTravelDuration, rewindSpeed, realCycleLength, realElapsed };
}

// Elapsed time (ms) within the current looping playback cycle, in the same
// units as playbackSchedule's offsetMs — shared by image-frame selection and
// camera keyframe animation so the two always stay in lockstep. During the
// start pause, elapsed is pushed far negative so nothing is highlighted yet
// (camera holds at keyframe 0). After the extended-forward phase, elapsed is
// pushed far past lastOffset so the last image's moment window (already
// closed naturally) stays closed for the end pause (screen blank, camera
// holds) instead of staying lit throughout.
function getPlaybackElapsedMs() {
  if (playbackSchedule.length === 0) return 0;

  const { lastOffset, realExtendedTravelDuration, rewindSpeed, realElapsed } = getPlaybackPhaseInfo();

  if (realElapsed < PLAYBACK_START_PAUSE_MS) {
    return -1e6; // before any moment window — blank, camera holds at keyframe 0
  }

  const realForwardElapsed = realElapsed - PLAYBACK_START_PAUSE_MS;

  if (realForwardElapsed < realExtendedTravelDuration) {
    return realForwardElapsed * PLAYBACK_SPEED;
  }

  if (realForwardElapsed < realExtendedTravelDuration + PLAYBACK_END_PAUSE_MS) {
    return lastOffset + 1e6; // far outside any moment window — blank, camera holds
  }

  const rewindReal = realForwardElapsed - realExtendedTravelDuration - PLAYBACK_END_PAUSE_MS;
  return lastOffset - rewindReal * rewindSpeed;
}

// True only during the rewind leg — used to suppress the per-image moment
// highlight there, since rewind is just the camera travelling back to the
// start rather than a second forward playthrough.
function isRewinding() {
  if (playbackSchedule.length === 0) return false;
  const { realExtendedTravelDuration, realElapsed } = getPlaybackPhaseInfo();
  const realForwardElapsed = realElapsed - PLAYBACK_START_PAUSE_MS;
  return realForwardElapsed >= realExtendedTravelDuration + PLAYBACK_END_PAUSE_MS;
}

// Alpha (0..1) for one image at the given elapsed time: LOW_ALPHA before its
// own offset, jumping instantly to full opacity right at that offset (no
// fade-in), holding at 1, then ramping back down to LOW_ALPHA over the last
// fadeWindow before holdWindow ends — never brightening before the image's
// own scheduled moment, only after.
function holdAlphaAt(elapsed, offsetMs, holdWindow, fadeWindow) {
  if (elapsed < offsetMs || elapsed >= offsetMs + holdWindow) return LOW_ALPHA;

  const t = elapsed - offsetMs;
  let frac = 1;
  if (t > holdWindow - fadeWindow) {
    frac = (holdWindow - t) / fadeWindow;
  }
  frac = constrain(frac, 0, 1);

  return LOW_ALPHA + (1 - LOW_ALPHA) * frac;
}

// Writes the current playback clock to a plain DOM element outside the
// canvas (#debug-time), so timing can be read directly off the page rather
// than inferred from what's rendered.
function updateDebugTimeDisplay(elapsed, highlightIndex) {
  const el = document.getElementById('debug-time');
  if (!el) return;

  const phase = isPaused ? 'paused' : (isRewinding() ? 'rewinding' : 'forward');
  const highlightLabel = highlightIndex >= 0 ? `#${highlightIndex}` : 'none';
  el.textContent = `elapsed: ${elapsed.toFixed(0)}ms | speed: ${PLAYBACK_SPEED}x | phase: ${phase} | highlighted: ${highlightLabel}`;
}

// stripShear lives in opencv-featurematch-js.js

// Fits a single image edge-to-edge ("square" to the camera) into a
// perspective camera's view, deriving size/center/roll directly from its own
// transformed corners (via the same applyTransform4x4 used to draw it) —
// an aligned image is often rotated slightly in world space, so an
// axis-aligned bounding box around it (viewed by a non-rolled camera) would
// leave gaps rather than exactly filling the frame.
function computeCameraKeyframeForImage(image, transform) {
  const w = image.naturalWidth || image.width;
  const h = image.naturalHeight || image.height;

  const [cx, cy] = applyTransform4x4(w / 2, h / 2, transform);
  const [tlx, tly] = applyTransform4x4(0, 0, transform);
  const [trx, tryy] = applyTransform4x4(w, 0, transform);
  const [blx, bly] = applyTransform4x4(0, h, transform);

  // Right/up edge vectors of the transformed image in world space — their
  // lengths give the image's true (un-inflated) footprint. Real (especially
  // weakly-matched) homographies carry real shear, so the left edge and top
  // edge don't rotate by quite the same angle — the camera roll is derived
  // from the top edge (rotated 90°) rather than the left edge directly, to
  // match isReasonableHomography's own atan2(c,a) rotation convention and
  // stay immune to the left edge's independent shear noise.
  const rightX = trx - tlx, rightY = tryy - tly;
  const worldW = Math.hypot(rightX, rightY);
  const worldH = Math.hypot(blx - tlx, bly - tly);
  const rightLen = worldW || 1;
  const up = [-rightY / rightLen, rightX / rightLen, 0];

  const fovY = PI / 3; // p5's default WEBGL vertical field of view (60deg)
  const aspect = width / height;

  // "Contain" fit, not "cover" — the whole image must stay visible with no
  // cropping; when its aspect ratio doesn't match the canvas's, the
  // less-restrictive dimension is left with empty margins instead.
  const distForHeight = (worldH / 2) / Math.tan(fovY / 2);
  const distForWidth = (worldW / 2) / (Math.tan(fovY / 2) * aspect);
  const dist = Math.max(distForHeight, distForWidth);

  return {
    eye: [cx, cy, dist],
    center: [cx, cy, 0],
    up
  };
}

// Derives one camera keyframe per successfully-aligned image (from the same
// transforms buildPlaybackSchedule just validated), framing that image alone.
// Each keyframe's time matches that image's playbackSchedule offset exactly,
// so the camera is precisely, squarely framed on an image the moment it's due.
function buildCameraKeyframes() {
  const mediaElement = select('#media')?.elt;
  if (!mediaElement || playbackSchedule.length === 0) { cameraKeyframes = []; return; }

  cameraKeyframes = playbackSchedule.map(entry => {
    const image = mediaElement.children[entry.index].querySelector('.original');
    const transform = stripShear(getImageTransformFromElement(image, true));
    const pose = computeCameraKeyframeForImage(image, transform);
    return { time: entry.offsetMs, ...pose };
  });
}

function lerp3(a, b, t) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t
  ];
}

// Returns the interpolated {eye, center, up} camera pose for the current
// moment: exactly kfA at kfA.time, exactly kfB at kfB.time, travelling smoothly
// between — using the same elapsed clock that gates each image's own moment
// window in draw(), so the camera is always centred on whichever image (if
// any) is currently in its moment.
function getCameraPose() {
  if (cameraKeyframes.length === 0) return null;
  if (cameraKeyframes.length === 1) return cameraKeyframes[0];

  const elapsed = getPlaybackElapsedMs();

  let i = 0;
  while (i < cameraKeyframes.length - 1 && cameraKeyframes[i + 1].time <= elapsed) i++;

  const kfA = cameraKeyframes[i];
  const kfB = cameraKeyframes[Math.min(i + 1, cameraKeyframes.length - 1)];
  if (kfA === kfB) return kfA;

  const span = kfB.time - kfA.time;
  const t = span > 0 ? constrain((elapsed - kfA.time) / span, 0, 1) : 1;

  return {
    eye: lerp3(kfA.eye, kfB.eye, t),
    center: lerp3(kfA.center, kfB.center, t),
    up: lerp3(kfA.up, kfB.up, t)
  };
}

// Reads DateTimeOriginal + SubSecTimeOriginal via exif-js and combines them
// into a single Date with millisecond precision (EXIF only stores whole seconds).
function extractImageTimestamp(imgElement) {
  return new Promise((resolve) => {
    EXIF.getData(imgElement, function () {
      const dateTimeOriginal = EXIF.getTag(this, 'DateTimeOriginal');
      const subSecTimeOriginal = EXIF.getTag(this, 'SubsecTimeOriginal');
      const timestamp = parseExifDateTime(dateTimeOriginal, subSecTimeOriginal);
      console.log('EXIF timestamp for', imgElement.src, ':', dateTimeOriginal, subSecTimeOriginal, '->', timestamp);
      resolve(timestamp);
    });
  });
}

function parseExifDateTime(dateTimeOriginal, subSecTimeOriginal) {
  if (!dateTimeOriginal) return null;

  const match = dateTimeOriginal.match(/^(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})$/);
  if (!match) return null;

  const [year, month, day, hour, minute, second] = match.slice(1).map(Number);
  const ms = subSecTimeOriginal ? Math.round(parseFloat('0.' + subSecTimeOriginal) * 1000) : 0;

  // EXIF months are 1-indexed; JS Date months are 0-indexed
  return new Date(year, month - 1, day, hour, minute, second, ms);
}

function setImageTimestamp(element, timestamp) {
  console.log('setImageTimestamp', element, timestamp);
  if (element && timestamp instanceof Date) {
    element.setAttribute('data-timestamp', timestamp.getTime());
  }
}

function getImageTimestampFromElement(element) {
  if (element) {
    const t = element.getAttribute('data-timestamp');
    if (t !== null) return new Date(Number(t));
  }
  return null;
}

async function processImage(originalImgElement, div) {
  // Already aligned (e.g. baked into the page's static HTML by a previous
  // run) — skip the segmentation/masking pipeline, since processHomography()
  // would just throw its output away via its own already-aligned check.
  // Trade-off: without a .background element, a pre-aligned frame like this
  // can no longer be used as a match candidate for any newly-dropped photo.
  if (getImageTransformFromElement(div)) return;

  // 1. Generate low-res image and wait for it to load
  const lowResImg = await generateLowResImageAsync(originalImgElement);
  lowResImg.parent(div);
  lowResImg.addClass('lowres');

  // 2. Generate mask and wait for it to load
  const maskImg = await generateMaskAsync(lowResImg.elt);
  maskImg.parent(div);
  maskImg.addClass('mask');

  // 3. Apply mask to get foreground and background, wait for both
  const [foregroundImg, backgroundImg] = await Promise.all([
    applyMaskToImageAsync(lowResImg.elt, maskImg.elt, false),
    applyMaskToImageAsync(lowResImg.elt, maskImg.elt, true)
  ]);
  foregroundImg.parent(div);
  foregroundImg.addClass('foreground');
  backgroundImg.parent(div);
  backgroundImg.addClass('background');
}