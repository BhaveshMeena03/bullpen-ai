// Records the token dashboard into frames, then ffmpeg stitches a GIF.
// Drives the LIVE page, so the ranking and timestamps are real output.
//
//   node scripts/make_gif5.mjs https://marketbubble-search.onrender.com/demo/assets.html
//
// The story it tells: the ranked list -> expand a token -> real timestamped
// moments you can click through to the exact second in the episode.

import { chromium } from "playwright-core";
import { mkdirSync, rmSync } from "fs";

const URL = process.argv[2] ||
  "https://marketbubble-search.onrender.com/demo/assets.html";
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const FRAMES = "/tmp/mb_token_frames";
const W = 1000, H = 760;

rmSync(FRAMES, { recursive: true, force: true });
mkdirSync(FRAMES, { recursive: true });

const browser = await chromium.launch({ executablePath: CHROME });
const page = await browser.newPage({
  viewport: { width: W, height: H },
  deviceScaleFactor: 2,
});
await page.goto(URL, { waitUntil: "networkidle" });
await page.waitForSelector(".row", { timeout: 30000 });
await page.waitForTimeout(600);

let n = 0;
const shoot = async (count = 1) => {
  for (let i = 0; i < count; i++) {
    await page.screenshot({
      path: `${FRAMES}/f${String(n++).padStart(4, "0")}.png`,
    });
  }
};

// 1. Hold on the headline + chart.
await shoot(10);

// 2. Scroll down to the ranked table.
for (let y = 0; y <= 620; y += 62) {
  await page.evaluate((v) => window.scrollTo(0, v), y);
  await page.waitForTimeout(60);
  await shoot(1);
}
await shoot(6);

// 3. Expand the top token -> real timestamped moments appear.
await page.evaluate(() => {
  document.querySelectorAll(".row")[0].querySelector(".head").click();
});
await page.waitForTimeout(350);
await shoot(14);

// 4. Drift down through the moments so the timestamps read.
for (let i = 0; i < 9; i++) {
  await page.evaluate(() => window.scrollBy(0, 58));
  await page.waitForTimeout(70);
  await shoot(1);
}
await shoot(12);

await browser.close();
console.log(`captured ${n} frames -> ${FRAMES}`);
