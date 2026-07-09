// identityMatrix, applyTransform4x4, multiplyMatrix4x4, invertMatrix4x4 and
// to2dAffine live in shimage.js; stripShear lives in opencv-featurematch-js.js
const imageTransforms = [];
let mediaBoundingBox = null;

// Once a candidate match clears this many RANSAC inliers, stop searching further
// (temporally-nearer) candidates — an early exit to avoid an O(n^2) alignment
// search across the whole sequence when the nearest neighbour is already a
// confident match, which is true for the large majority of burst-sequence pairs.
const EARLY_EXIT_INLIER_THRESHOLD = 50;

// Vertical FOV computeCameraKeyframeForImage()'s "contain" fit distance math
// assumes, and what processAnyAttachedMedia() explicitly sets via
// perspective() after resizing the canvas. Must stay in sync between the
// two - p5's own default WEBGL FOV is derived from canvas height
// (2*atan(height/2/800)), so it silently drifts away from a fixed 60deg
// whenever the canvas isn't 800px tall, breaking the framing.
const CAMERA_FOV_Y = Math.PI / 3; // 60deg. Math.PI (not p5's PI global, unavailable at parse time)

let maskSegmentation = null;

// Real-time playback of the capture sequence, driven by each image's EXIF timestamp.
// Both 0: no held-still pause at either end - forward playback begins the
// instant rewind arrives back at the first image, and rewind begins the
// instant the last image's own hold ends, so the whole thing just loops
// continuously.
const PLAYBACK_START_PAUSE_MS = 0;
const PLAYBACK_END_PAUSE_MS = 0;
const PLAYBACK_SPEED = 0.5; // 1 = real-time (matches original capture pace), 0.5 = half speed
let playbackSchedule = [];
let playbackStartMillis = 0;

// The constant background level shown outside any image's own hold window.
const LOW_ALPHA = 0.1;

// The single image currently shown (only one is ever drawn - see draw()).
// Persists (rather than reverting to none) once its hold/fade ends, until
// the next image's hold begins and takes over.
let currentDisplayIndex = -1;
let currentDisplayAlpha = LOW_ALPHA;

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
const HOLD_MS = 350;

// Alpha jumps instantly to full opacity when a photo's hold begins, then
// ramps smoothly back down to LOW_ALPHA over this much of the end of the
// hold window (real ms, same convention as HOLD_MS). Clamped to the hold
// window's own length if HOLD_MS is shorter.
const FADE_MS = 150;

// 3D camera fly-through: one keyframe per aligned image, framing that image
// alone, timed to the exact same clock as playbackSchedule above — the camera
// is exactly centred on an image at the moment it becomes the current frame,
// then travels to be exactly framed on the next image by its due time.
let cameraKeyframes = [];

// A single straight line fit through every keyframe's eye position (see
// fitLineToPoints() in buildCameraKeyframes()) - the camera's position
// travels along this one line for the whole sequence (forward and rewind),
// rather than curving out to each photo's own ideal eye position.
let cameraPathLine = null;

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

// Canvas width once resized to a single photo's own aspect ratio - see
// processAnyAttachedMedia(), which resizes as soon as image dimensions are
// known. Used here only as setup()'s placeholder size before that happens.
const CANVAS_WIDTH = 800;

// In setup(), use async/await to block until ready
async function setup() {
  // Use WEBGL so texture()/vertex(u,v) in drawImageWithHomography works
  canvas = createCanvas(CANVAS_WIDTH, CANVAS_WIDTH, WEBGL);
  canvas.parent('canvas-target'); // explicit target, rather than p5's default append-to-body
  frameRate(60);
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
    const image = mediaElement.children[i].querySelector(selector);
    if (!image) continue;

    // Composed through the image itself (not just its parent div) - an
    // image can carry its own scale transform (e.g. a downsized .original
    // whose div-level alignment was computed at a different resolution),
    // which the div's own transform alone wouldn't include.
    const transform = getImageTransformFromElement(image, true);
    if (!transform) continue;

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

  const elapsed = getPlaybackElapsedMs();
  const phase = getPlaybackPhase();
  const holdWindow = HOLD_MS * PLAYBACK_SPEED;
  const fadeWindow = Math.min(FADE_MS * PLAYBACK_SPEED, holdWindow);

  // Exactly one image is shown at a time. It holds at full opacity on the
  // first image throughout the start pause, and on the last image
  // throughout the end pause, rather than the screen going blank there.
  // During forward playback, each image follows its own hold/fade window in
  // turn (holdAlphaAt); the most recently active one keeps showing at
  // LOW_ALPHA until the next image's window begins. Rewind instead just
  // shows whichever image is nearest the current (backwards-running)
  // elapsed time - see the 'rewind' branch below for why.
  let highlightIndex = -1;
  let highlightAlpha = LOW_ALPHA;

  if (phase === 'start-pause') {
    highlightIndex = playbackSchedule[0].index;
    highlightAlpha = 1;
  } else if (phase === 'end-pause') {
    highlightIndex = playbackSchedule[playbackSchedule.length - 1].index;
    highlightAlpha = 1;
  } else if (phase === 'forward') {
    let bestPos = 0;
    let bestAlpha = -Infinity;
    for (let p = 0; p < playbackSchedule.length; p++) {
      const a = holdAlphaAt(elapsed, playbackSchedule[p].offsetMs, holdWindow, fadeWindow);
      if (a > bestAlpha) { bestAlpha = a; bestPos = p; }
    }
    if (bestAlpha > LOW_ALPHA) {
      highlightIndex = playbackSchedule[bestPos].index;
      highlightAlpha = bestAlpha;
    }
  } else if (phase === 'rewind') {
    // holdAlphaAt's hold/fade window is derived from real time (HOLD_MS *
    // PLAYBACK_SPEED), but the fixed REWIND_DURATION_MS rewind sweeps
    // through the whole capture-time range far faster - most images'
    // windows end up only a few ms wide and never land on an actual 10fps
    // frame. Show whichever image's own offset is nearest instead, so every
    // image gets its own real-time slice of the rewind, however brief.
    let bestPos = 0;
    let bestDist = Infinity;
    for (let p = 0; p < playbackSchedule.length; p++) {
      const d = Math.abs(playbackSchedule[p].offsetMs - elapsed);
      if (d < bestDist) { bestDist = d; bestPos = p; }
    }
    highlightIndex = playbackSchedule[bestPos].index;
    highlightAlpha = LOW_ALPHA;
  }

  if (highlightIndex >= 0) {
    currentDisplayIndex = highlightIndex;
    currentDisplayAlpha = highlightAlpha;
  } else if (currentDisplayIndex >= 0) {
    currentDisplayAlpha = LOW_ALPHA;
  }

  updateDebugTimeDisplay(elapsed, highlightIndex);

  // The camera stays parked on each image's own keyframe throughout that
  // image's hold/fade window (see getCameraPose()), only travelling to the
  // next keyframe once that window ends - it doesn't pan away mid-burst.
  const camPose = getCameraPose();
  if (camPose) {
    camera(
      camPose.eye[0], camPose.eye[1], camPose.eye[2],
      camPose.center[0], camPose.center[1], camPose.center[2],
      camPose.up[0], camPose.up[1], camPose.up[2]
    );
  }

  const mediaElement = select('#media')?.elt;
  if (mediaElement && currentDisplayIndex >= 0) {
    const imageEl = mediaElement.children[currentDisplayIndex].querySelector(imageSelector);
    if (imageEl) {
      // applyMatrix() + image() lets the GPU's own model-matrix stack (and
      // its automatic perspective divide) do the projection, instead of
      // drawProjectedImage()'s CPU-side per-corner applyTransform4x4() into
      // a hand-built textured quad - see
      // https://editor.p5js.org/davidchatting/sketches/YHF4dsSbR. Still
      // goes through getTextureFromElement() (shimage.js) rather than
      // image()'ing the raw <img> directly, since a p5.Graphics is a
      // proven-compatible WEBGL image() source and a raw DOM element isn't.
      // (Named imageEl, not image, so it doesn't shadow p5's image()
      // function within this scope.)
      // No stripShear() here (unlike computeCameraKeyframeForImage, which
      // still uses it for shear-immune camera framing) - the real
      // homography's shear/perspective is now rendered as-is, since the
      // GPU's own perspective divide handles it natively.
      push();
        tint(255, 255 * currentDisplayAlpha);
        const t = getImageTransformFromElement(imageEl, true);
        applyMatrix(t);
        image(getTextureFromElement(imageEl), 0, 0);
      pop();
    }
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

  // Now that a photo's natural dimensions are known, size the canvas to
  // that single photo's own aspect ratio at CANVAS_WIDTH, rather than
  // setup()'s square placeholder. Any one photo stands in for all of them -
  // they're a burst sequence from one camera, so all share the same ratio.
  if (originals.length > 0) {
    const sample = originals[0].elt;
    const w = sample.naturalWidth || sample.width;
    const h = sample.naturalHeight || sample.height;
    if (w && h) {
      const canvasHeight = Math.round(CANVAS_WIDTH * (h / w));
      resizeCanvas(CANVAS_WIDTH, canvasHeight);

      // p5's default WEBGL perspective silently changes with canvas height
      // (see CAMERA_FOV_Y above) - fix it explicitly to match what
      // computeCameraKeyframeForImage()'s distance math assumes. near/far
      // are generous fixed bounds; photo-pixel-space world units run into
      // the thousands, well within [1, 1e6].
      perspective(CAMERA_FOV_Y, CANVAS_WIDTH / canvasHeight, 1, 1e6);
    }
  }

  // Phase 1: recover every image's EXIF timestamp up front, before any
  // alignment work — the processing order below (and buildPlaybackSchedule
  // afterwards) is derived from these.
  for (const orig of originals) {
    // Only default to identity if nothing's set yet - a hydrated data-key
    // can give an .original its own real scale-compensation transform (see
    // getBoundingBox()'s comment above), which this would otherwise
    // clobber back to identity every load.
    if (!getImageTransformFromElement(orig.elt)) {
      setImageTransform(orig.elt, identityMatrix);
    }
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

// Rewind always takes exactly this long in real wall-clock time, regardless
// of the burst's total duration (lastOffset) — a quick single motion rather
// than a speed-scaled mirror of the forward pass.
const REWIND_DURATION_MS = 1000;

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
//  5. Rewind: lastOffset -> 0, always over exactly REWIND_DURATION_MS of
//     real time (rewindSpeed is derived from lastOffset so the whole
//     timeline fits in that fixed duration, however long the burst is).
function getPlaybackPhaseInfo() {
  const lastOffset = playbackSchedule[playbackSchedule.length - 1].offsetMs;
  const realTravelDuration = lastOffset / PLAYBACK_SPEED;
  const realExtendedTravelDuration = realTravelDuration + HOLD_MS;
  const realRewindDuration = REWIND_DURATION_MS;
  const rewindSpeed = lastOffset / realRewindDuration;
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

// One of 'start-pause' | 'forward' | 'end-pause' | 'rewind' — same phase
// boundaries as getPlaybackElapsedMs()/isRewinding(), exposed directly so
// draw() can hold on the first image throughout the start pause and the
// last image throughout the end pause.
function getPlaybackPhase() {
  if (playbackSchedule.length === 0) return 'forward';

  const { realExtendedTravelDuration, realElapsed } = getPlaybackPhaseInfo();
  if (realElapsed < PLAYBACK_START_PAUSE_MS) return 'start-pause';

  const realForwardElapsed = realElapsed - PLAYBACK_START_PAUSE_MS;
  if (realForwardElapsed < realExtendedTravelDuration) return 'forward';
  if (realForwardElapsed < realExtendedTravelDuration + PLAYBACK_END_PAUSE_MS) return 'end-pause';
  return 'rewind';
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

  const fovY = CAMERA_FOV_Y;
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
// Least-squares straight line through a set of 3D points: origin is their
// centroid, dir is the direction of maximum variance (the dominant
// eigenvector of their covariance matrix), found via power iteration - a
// simple, dependency-free way to get it for the handful of points a burst
// sequence has, without needing a full eigendecomposition library. The
// direction's sign is arbitrary (power iteration can settle on either
// +v or -v), but that's fine - projectOntoLine() is always used
// self-consistently against whichever sign this happens to converge to.
function fitLineToPoints(points) {
  const n = points.length;
  const origin = [0, 0, 0];
  for (const p of points) { origin[0] += p[0]; origin[1] += p[1]; origin[2] += p[2]; }
  origin[0] /= n; origin[1] /= n; origin[2] /= n;

  // 3x3 covariance matrix of the centred points
  const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const p of points) {
    const d = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        cov[r][c] += d[r] * d[c];
      }
    }
  }

  let dir = [1, 1, 1];
  for (let iter = 0; iter < 100; iter++) {
    const next = [
      cov[0][0] * dir[0] + cov[0][1] * dir[1] + cov[0][2] * dir[2],
      cov[1][0] * dir[0] + cov[1][1] * dir[1] + cov[1][2] * dir[2],
      cov[2][0] * dir[0] + cov[2][1] * dir[1] + cov[2][2] * dir[2]
    ];
    const len = Math.hypot(next[0], next[1], next[2]) || 1;
    dir = [next[0] / len, next[1] / len, next[2] / len];
  }

  return { origin, dir };
}

// Scalar distance (in dir's units, i.e. already normalized) of point's
// projection onto the line from origin along dir - reconstruct the actual
// point via origin + dir*s.
function projectOntoLine(point, origin, dir) {
  return (point[0] - origin[0]) * dir[0] + (point[1] - origin[1]) * dir[1] + (point[2] - origin[2]) * dir[2];
}

function buildCameraKeyframes() {
  const mediaElement = select('#media')?.elt;
  if (!mediaElement || playbackSchedule.length === 0) { cameraKeyframes = []; cameraPathLine = null; return; }

  cameraKeyframes = playbackSchedule.map(entry => {
    const image = mediaElement.children[entry.index].querySelector('.original');
    const transform = stripShear(getImageTransformFromElement(image, true));
    const pose = computeCameraKeyframeForImage(image, transform);
    return { time: entry.offsetMs, ...pose };
  });

  cameraPathLine = fitLineToPoints(cameraKeyframes.map(kf => kf.eye));
  for (const kf of cameraKeyframes) {
    kf.eyeLineParam = projectOntoLine(kf.eye, cameraPathLine.origin, cameraPathLine.dir);
  }
}

function lerp3(a, b, t) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t
  ];
}

// Standard (uniform) Catmull-Rom spline through p1->p2, using p0/p3 as the
// neighbouring control points that shape the curve's tangents. Passes
// exactly through p1 at t=0 and p2 at t=1, same as lerp(p1, p2, t) would -
// only the path between them curves instead of following a straight line.
function catmullRom1D(p0, p1, p2, p3, t) {
  const t2 = t * t;
  const t3 = t2 * t;
  return 0.5 * (
    (2 * p1) +
    (-p0 + p2) * t +
    (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
    (-p0 + 3 * p1 - 3 * p2 + p3) * t3
  );
}

function catmullRom3(p0, p1, p2, p3, t) {
  return [
    catmullRom1D(p0[0], p1[0], p2[0], p3[0], t),
    catmullRom1D(p0[1], p1[1], p2[1], p3[1], t),
    catmullRom1D(p0[2], p1[2], p2[2], p3[2], t)
  ];
}

// Returns the interpolated {eye, center, up} camera pose for the current
// moment. eye always travels along cameraPathLine, the single straight line
// fitted across every keyframe's position - forward or rewind, the camera
// never leaves that line. center/up (look direction) still curve via
// Catmull-Rom through the surrounding keyframes during forward travel, or
// lerp3 directly during rewind. Forward: exactly kfA from kfA.time through
// kfA's own hold/fade window (the camera stays parked there for the full
// "burst" rather than immediately setting off for kfB), then travels the
// remaining gap, arriving exactly at kfB by kfB.time. Rewind: straight from
// the last keyframe back to the first, over REWIND_DURATION_MS - no need to
// retrace every intermediate keyframe on the way back. Uses the same
// elapsed clock that gates each image's own moment window in draw().
function getCameraPose() {
  if (cameraKeyframes.length === 0) return null;
  if (cameraKeyframes.length === 1) return cameraKeyframes[0];

  const elapsed = getPlaybackElapsedMs();

  if (isRewinding()) {
    const first = cameraKeyframes[0];
    const last = cameraKeyframes[cameraKeyframes.length - 1];
    const lastOffset = playbackSchedule[playbackSchedule.length - 1].offsetMs;
    const t = lastOffset > 0 ? constrain(1 - elapsed / lastOffset, 0, 1) : 1;

    return {
      eye: eyeOnPathLine(lerp1(last.eyeLineParam, first.eyeLineParam, t)),
      center: lerp3(last.center, first.center, t),
      up: lerp3(last.up, first.up, t)
    };
  }

  let i = 0;
  while (i < cameraKeyframes.length - 1 && cameraKeyframes[i + 1].time <= elapsed) i++;

  const kfA = cameraKeyframes[i];
  const kfB = cameraKeyframes[Math.min(i + 1, cameraKeyframes.length - 1)];
  if (kfA === kfB) return kfA;

  const holdWindow = HOLD_MS * PLAYBACK_SPEED;
  const departTime = kfA.time + holdWindow;
  const span = kfB.time - departTime;
  const t = elapsed <= departTime ? 0 : (span > 0 ? constrain((elapsed - departTime) / span, 0, 1) : 1);

  // The stop/travel timing above is unchanged. eye now travels along the
  // single straight line fitted across every keyframe (cameraPathLine, see
  // buildCameraKeyframes()) rather than curving out to each photo's own
  // ideal position - just linear interpolation of each keyframe's scalar
  // position along that line. center/up still curve via Catmull-Rom
  // through the surrounding keyframes, since they're look-direction, not
  // position.
  const kfPrev = cameraKeyframes[Math.max(i - 1, 0)];
  const kfNext = cameraKeyframes[Math.min(i + 2, cameraKeyframes.length - 1)];

  return {
    eye: eyeOnPathLine(lerp1(kfA.eyeLineParam, kfB.eyeLineParam, t)),
    center: catmullRom3(kfPrev.center, kfA.center, kfB.center, kfNext.center, t),
    up: catmullRom3(kfPrev.up, kfA.up, kfB.up, kfNext.up, t)
  };
}

function lerp1(a, b, t) {
  return a + (b - a) * t;
}

function eyeOnPathLine(s) {
  const { origin, dir } = cameraPathLine;
  return [origin[0] + dir[0] * s, origin[1] + dir[1] * s, origin[2] + dir[2] * s];
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