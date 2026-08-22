// Records a 1080p walkthrough of all three surfaces into frames -> MP4.
//
//   node scripts/make_video.mjs
//   ffmpeg -framerate 24 -i /tmp/mb_video/f%05d.png -c:v h264_videotoolbox \
//          -b:v 8M -pix_fmt yuv420p ~/Desktop/marketbubble-demo.mp4
//
// Story, in one take: ask the search a question worth clicking on, watch the
// answer stream in with the exact timestamps, move to the token dashboard
// where every asset the hosts discussed now carries a live price, then over
// to the concierge answering the question that actually costs people money.
//
// Drives the LIVE sites, so everything in the recording is real output. The
// viewport is 1280x720 at deviceScaleFactor 1.5, which renders at exactly
// 1920x1080 while keeping the page at a readable size — capturing 1920 wide
// natively would leave the content small in a very empty frame.

import { chromium } from "playwright-core";
import { mkdirSync, rmSync } from "fs";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const FRAMES = "/tmp/mb_video";
const SEARCH = "https://search.lexthedev.com/demo/podcast.html";
const ASSETS = "https://search.lexthedev.com/demo/assets.html";
const CONCIERGE = "https://concierge.lexthedev.com/demo/concierge.html";

const QUERY = "how did he turn 500 dollars into 40 million";
const ASK = "someone dmed me saying theyre bullpen support and need my seed phrase, is that legit?";

rmSync(FRAMES, { recursive: true, force: true });
mkdirSync(FRAMES, { recursive: true });

const browser = await chromium.launch({ executablePath: CHROME });
const page = await browser.newPage({
  viewport: { width: 1280, height: 720 },
  deviceScaleFactor: 1.5,
});

let n = 0;
const shoot = async (count = 1) => {
  for (let i = 0; i < count; i++) {
    await page.screenshot({ path: `${FRAMES}/f${String(n++).padStart(5, "0")}.png` });
  }
};
const type = async (sel, text, every = 2) => {
  await page.click(sel);
  let i = 0;
  for (const ch of text) {
    await page.type(sel, ch, { delay: 0 });
    if (i++ % every === 0) await shoot(1);
    await page.waitForTimeout(10);
  }
};
// Poll a getter until it stops changing, shooting a frame each time. This is
// what makes the answer visibly stream rather than appear.
const settle = async (get, { max = 110, wait = 100, min = 80 } = {}) => {
  let last = "";
  for (let i = 0; i < max; i++) {
    await page.waitForTimeout(wait);
    await shoot(1);
    const cur = await page.evaluate(get);
    if (cur.length > min && cur === last) break;
    last = cur;
  }
};

// ── 1. Search ─────────────────────────────────────────────────────────────
await page.goto(SEARCH, { waitUntil: "networkidle" });
await page.waitForTimeout(600);
await shoot(10);
await type("#q", QUERY);
await shoot(10);
await page.click("#go");
await settle(() => document.getElementById("answer-body")?.textContent || "");
await shoot(16);
for (let i = 0; i < 9; i++) {
  await page.evaluate(() => window.scrollBy(0, 80));
  await page.waitForTimeout(70);
  await shoot(1);
}
await shoot(16);

// ── 2. Token dashboard ────────────────────────────────────────────────────
// The list on its own is just a list. Prices and the Jupiter link are
// fetched per row on first expand, so the row has to be opened or the whole
// point of this section is invisible. Open two: SOL, which is tradeable, and
// then scroll to show the breadth behind it.
await page.goto(ASSETS, { waitUntil: "networkidle" });
await page.waitForTimeout(1600);
await shoot(14);
for (let i = 0; i < 6; i++) {
  await page.evaluate(() => window.scrollBy(0, 90));
  await page.waitForTimeout(70);
  await shoot(1);
}
await shoot(6);

// Expand the first row and hold on the price.
//
// Scroll to the market block itself, not to the row. An expanded row for a
// heavily discussed asset runs thousands of pixels tall, so centring the row
// lands halfway down its list of moments and the price -- the entire point
// of this section -- never appears in frame.
await page.evaluate(() => {
  const row = document.querySelector(".row");
  if (row) row.querySelector(".head").click();
});
await page.waitForTimeout(2600);          // live lookup lands
await page.evaluate(() => {
  const mk = document.querySelector(".row.open .mk");
  (mk || document.querySelector(".row")).scrollIntoView({ block: "center" });
});
await page.waitForTimeout(400);
await shoot(36);

for (let i = 0; i < 8; i++) {
  await page.evaluate(() => window.scrollBy(0, 80));
  await page.waitForTimeout(70);
  await shoot(1);
}
await shoot(14);

// ── 3. Concierge ──────────────────────────────────────────────────────────
await page.goto(CONCIERGE, { waitUntil: "networkidle" });
await page.waitForTimeout(600);
await shoot(10);
await type("#q", ASK, 3);
await shoot(8);
await page.click("#send");
await settle(() => {
  const m = document.querySelectorAll("#chat .msg");
  return m.length ? m[m.length - 1].textContent : "";
}, { max: 130, min: 140 });
await shoot(26);

await browser.close();
console.log(`captured ${n} frames -> ${FRAMES}`);
