// Records the one thing nobody else can show: the same search reaching
// BOTH the YouTube uploads and the live X broadcasts — with a caption on
// screen saying what is happening at each step.
//
//   node scripts/make_platform_demo.mjs
//   ffmpeg -framerate 24 -i /tmp/mb_platform/f%05d.png \
//          -c:v h264_videotoolbox -b:v 8M -pix_fmt yuv420p \
//          ~/Desktop/marketbubble-x-and-youtube.mp4
//
// Two queries, chosen because they land on different platforms, and both
// verified against the live index before this was written:
//
//   "how did he turn 500 dollars into 40 million"  -> YouTube, 18:07, and
//        the link jumps to that second.
//   "what did jesse pollak say about base"         -> a live X broadcast
//        at 56:47 — material that never reaches the YouTube upload, which
//        is the part of the archive that exists nowhere else.
//
// Drives the live site, so every answer and citation is real output rather
// than a mock. If the model phrases an answer differently on the day the
// recording is still true, which is the point of recording the real thing.

import { chromium } from "playwright-core";
import { mkdirSync, rmSync } from "fs";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const FRAMES = "/tmp/mb_platform";
const SEARCH = "https://search.lexthedev.com/demo/podcast.html";

const ON_YOUTUBE = "how did he turn 500 dollars into 40 million";
const ON_X = "what did jesse pollak say about base";

rmSync(FRAMES, { recursive: true, force: true });
mkdirSync(FRAMES, { recursive: true });

const browser = await chromium.launch({ executablePath: CHROME });
const page = await browser.newPage({
  viewport: { width: 1280, height: 720 },
  deviceScaleFactor: 1.5,          // renders at exactly 1920x1080
});

let n = 0;
const shoot = async (count = 1) => {
  for (let i = 0; i < count; i++) {
    await page.screenshot({
      path: `${FRAMES}/f${String(n++).padStart(5, "0")}.png`,
    });
  }
};

// The caption bar. Injected rather than burned in afterwards, so it moves
// with the page and needs no video editor. Sits at the bottom because the
// answer builds from the top and covering that would defeat the recording.
const say = async (text) => {
  await page.evaluate((message) => {
    let bar = document.getElementById("__caption");
    if (!bar) {
      bar = document.createElement("div");
      bar.id = "__caption";
      bar.style.cssText = [
        "position:fixed", "left:0", "right:0", "bottom:0", "z-index:99999",
        "padding:18px 28px", "background:rgba(6,10,8,0.94)",
        "border-top:2px solid rgba(52,211,153,0.55)",
        "font:600 21px/1.4 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif",
        "color:#e8fff4", "letter-spacing:0.1px",
        "box-shadow:0 -10px 30px rgba(0,0,0,0.55)",
      ].join(";");
      document.body.appendChild(bar);
    }
    bar.textContent = message;
  }, text);
};

const clearCaption = async () => {
  await page.evaluate(() => document.getElementById("__caption")?.remove());
};

const type = async (text, every = 2) => {
  await page.click("#q");
  await page.fill("#q", "");
  let i = 0;
  for (const ch of text) {
    await page.type("#q", ch, { delay: 0 });
    if (i++ % every === 0) await shoot(1);
    await page.waitForTimeout(10);
  }
};

// Poll until the answer stops growing, shooting as it goes, so it streams
// on screen instead of cutting from empty to full.
const settle = async ({ max = 120, wait = 100, min = 80 } = {}) => {
  let last = "";
  for (let i = 0; i < max; i++) {
    await page.waitForTimeout(wait);
    await shoot(1);
    const now = await page.evaluate(
      () => document.getElementById("answer-body")?.textContent || "");
    if (now.length > min && now === last) break;
    last = now;
  }
};

const scroll = async (steps, by = 80) => {
  for (let i = 0; i < steps; i++) {
    await page.evaluate((d) => window.scrollBy(0, d), by);
    await page.waitForTimeout(70);
    await shoot(1);
  }
};

const ask = async (question, captions) => {
  await page.goto(SEARCH, { waitUntil: "networkidle" });
  await page.waitForTimeout(700);
  await clearCaption();
  await say(captions.asking);
  await shoot(16);
  await type(question);
  await shoot(14);
  await page.click("#go");
  await say(captions.answering);
  await settle();
  await shoot(24);
  await say(captions.cited);
  await shoot(16);
  await page.evaluate(() => {
    document.getElementById("hits")?.scrollIntoView({ block: "start" });
  });
  await page.waitForTimeout(400);
  await say(captions.sources);
  await shoot(34);
  await scroll(8);
  await shoot(22);
};

// 1 — a question the uploads answer, where the link lands on the second.
await ask(ON_YOUTUBE, {
  asking: "Ask in plain English. No keywords, no episode number.",
  answering: "The answer is written only from the transcripts.",
  cited: "Every claim carries the exact second it was said.",
  sources: "Each source jumps straight to that moment on YouTube.",
});

// 2 — a question only the live broadcast answers.
await ask(ON_X, {
  asking: "Now something that was only said on the live X broadcast.",
  answering: "Same search. The live streams are indexed too.",
  cited: "About a third of every show never reaches the YouTube upload.",
  sources: "That part is searchable here and nowhere else.",
});

await clearCaption();
await page.goto(SEARCH, { waitUntil: "networkidle" });
await page.waitForTimeout(600);
await say("search.lexthedev.com  ·  or tag @mbubbleSearch on X");
await shoot(40);

await browser.close();
console.log(`captured ${n} frames -> ${FRAMES}`);
console.log(
  "ffmpeg -framerate 24 -i /tmp/mb_platform/f%05d.png " +
  "-c:v h264_videotoolbox -b:v 8M -pix_fmt yuv420p " +
  "~/Desktop/marketbubble-x-and-youtube.mp4");
