// Market Bubble search demo GIF: two real searches over the indexed
// catalog, showing the answer + timestamp deep-link cards.
//
//   node scripts/make_gif3.mjs http://localhost:8100/demo/podcast.html

import { chromium } from "playwright-core";
import { mkdirSync, rmSync } from "fs";

const URL = process.argv[2] || "http://localhost:8100/demo/podcast.html";
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const FRAMES = "/tmp/bpc_frames3";
const W = 760, H = 1000;

rmSync(FRAMES, { recursive: true, force: true });
mkdirSync(FRAMES, { recursive: true });

const browser = await chromium.launch({ executablePath: CHROME });
const page = await browser.newPage({
  viewport: { width: W, height: H },
  deviceScaleFactor: 2,
});
await page.goto(URL, { waitUntil: "networkidle" });

let frame = 0;
let busy = false;
const ticker = setInterval(async () => {
  if (busy) return;
  busy = true;
  await page.screenshot({
    path: `${FRAMES}/f${String(frame++).padStart(4, "0")}.png`,
  }).catch(() => {});
  busy = false;
}, 350);

const hold = (ms) => page.waitForTimeout(ms);

async function search(q) {
  await page.fill("#q", "");
  // Type visibly so the GIF shows the question being asked.
  await page.type("#q", q, { delay: 28 });
  await page.click("#go");
  await page.waitForFunction(
    () => document.getElementById("answer").classList.contains("show"),
    { timeout: 60000 }
  ).catch(() => {});
  await hold(600);
  // Scroll the timestamp cards into view — they're the point.
  await page.evaluate(() => {
    document.getElementById("hits")?.scrollIntoView({ behavior: "smooth", block: "end" });
  });
  await hold(2600);
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));
  await hold(600);
}

await hold(1400);
await search("why does ansem think ethereum is done?");
await search("what did they say about robot stocks?");
await hold(1500);

clearInterval(ticker);
await browser.close();
console.log(`captured ${frame} frames -> ${FRAMES}`);
