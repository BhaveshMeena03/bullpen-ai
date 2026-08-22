// Records the concierge answering a real scam question into frames -> GIF.
// Drives the LIVE page, so the streamed answer and the channel list are
// genuine output rather than a mock.
//
//   node scripts/make_gif8.mjs
//
// Story: a user asks whether a "support" DM demanding their seed phrase is
// legit -> the answer streams in -> it names the red flags and gives only
// the verified channels.

import { chromium } from "playwright-core";
import { mkdirSync, rmSync } from "fs";

const URL = "https://concierge.lexthedev.com/demo/concierge.html";
const QUERY = "someone dmed me saying theyre bullpen support and need my seed phrase to unlock my account, is that legit?";
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const FRAMES = "/tmp/bp_concierge_frames";
const W = 1000, H = 820;

rmSync(FRAMES, { recursive: true, force: true });
mkdirSync(FRAMES, { recursive: true });

const browser = await chromium.launch({ executablePath: CHROME });
const page = await browser.newPage({
  viewport: { width: W, height: H },
  deviceScaleFactor: 2,
});
await page.goto(URL, { waitUntil: "networkidle" });
await page.waitForTimeout(600);

let n = 0;
const shoot = async (count = 1) => {
  for (let i = 0; i < count; i++) {
    await page.screenshot({ path: `${FRAMES}/f${String(n++).padStart(4, "0")}.png` });
  }
};

// 1. Empty state, then type the question.
await shoot(6);
await page.click("#q");
for (const ch of QUERY) {
  await page.type("#q", ch, { delay: 0 });
  await shoot(1);
  await page.waitForTimeout(14);
}
await shoot(8);

// 2. Send it; capture the answer streaming in.
await page.click("#send");
let last = "";
for (let i = 0; i < 120; i++) {
  await page.waitForTimeout(110);
  await shoot(1);
  const cur = await page.evaluate(() => {
    const msgs = document.querySelectorAll("#chat .msg");
    return msgs.length ? msgs[msgs.length - 1].textContent : "";
  });
  if (cur.length > 120 && cur === last) break;   // settled
  last = cur;
}
await shoot(14);

await browser.close();
console.log(`captured ${n} frames -> ${FRAMES}`);
