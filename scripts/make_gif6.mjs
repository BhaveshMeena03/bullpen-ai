// Records the search page answering a real question into frames -> GIF.
// Drives the LIVE page so the streamed answer and timestamps are real output.
//
//   node scripts/make_gif6.mjs
//
// Story: type a question -> the answer streams in from the transcripts ->
// timestamped source cards appear, each a link to the exact moment.

import { chromium } from "playwright-core";
import { mkdirSync, rmSync } from "fs";

const URL = "https://marketbubble-search.onrender.com/demo/podcast.html";
const QUERY = "what does Ansem look for in a good trade?";
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const FRAMES = "/tmp/mb_search_frames";
const W = 1000, H = 780;

rmSync(FRAMES, { recursive: true, force: true });
mkdirSync(FRAMES, { recursive: true });

const browser = await chromium.launch({ executablePath: CHROME });
const page = await browser.newPage({
  viewport: { width: W, height: H },
  deviceScaleFactor: 2,
});
await page.goto(URL, { waitUntil: "networkidle" });
await page.waitForTimeout(500);

let n = 0;
const shoot = async (count = 1) => {
  for (let i = 0; i < count; i++) {
    await page.screenshot({
      path: `${FRAMES}/f${String(n++).padStart(4, "0")}.png`,
    });
  }
};

// 1. Empty state, then type the question character by character.
await shoot(6);
await page.click("#q");
for (const ch of QUERY) {
  await page.type("#q", ch, { delay: 0 });
  if (n % 1 === 0) await shoot(1);
  await page.waitForTimeout(24);
}
await shoot(6);

// 2. Fire the search; capture the answer streaming + hits rendering.
await page.click("#go");
// Poll for streamed text growth so the GIF actually shows it filling in.
let last = "";
for (let i = 0; i < 90; i++) {
  await page.waitForTimeout(120);
  await shoot(1);
  const cur = await page.evaluate(() =>
    (document.getElementById("answer-body")?.textContent || ""));
  const hits = await page.evaluate(() =>
    document.querySelectorAll("#hits .hit").length);
  if (cur.length > 40 && cur === last && hits > 0) break;  // settled
  last = cur;
}
await shoot(10);

// 3. Scroll down to reveal the timestamped source cards.
for (let i = 0; i < 7; i++) {
  await page.evaluate(() => window.scrollBy(0, 74));
  await page.waitForTimeout(80);
  await shoot(1);
}
await shoot(12);

await browser.close();
console.log(`captured ${n} frames -> ${FRAMES}`);
