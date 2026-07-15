// Records the Bullpen Concierge demo into frames, then ffmpeg stitches a GIF.
// Drives the LIVE widget so the captured answers are real model output.
//
//   node scripts/make_gif.mjs http://localhost:8100/demo/
//
// Frames are captured on an interval while a scripted conversation plays,
// so the GIF shows text streaming in — not just before/after stills.

import { chromium } from "playwright-core";
import { mkdirSync, rmSync } from "fs";

const URL = process.argv[2] || "http://localhost:8100/demo/";
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const FRAMES = "/tmp/bpc_frames";
const W = 900, H = 1200;

rmSync(FRAMES, { recursive: true, force: true });
mkdirSync(FRAMES, { recursive: true });

const browser = await chromium.launch({ executablePath: CHROME });
const page = await browser.newPage({
  viewport: { width: W, height: H },
  deviceScaleFactor: 2,
});
await page.goto(URL, { waitUntil: "networkidle" });

let frame = 0;
let capturing = false;
const snap = async () => {
  if (capturing) return;
  capturing = true;
  await page.screenshot({
    path: `${FRAMES}/f${String(frame++).padStart(4, "0")}.png`,
  });
  capturing = false;
};
// Capture ~3 fps throughout so streaming text is visible.
const ticker = setInterval(snap, 330);

const hold = (ms) => page.waitForTimeout(ms);

async function ask(text) {
  await page.fill(".bpc-input", text);
  await page.click(".bpc-send");
  // Wait for the send button to re-enable (answer complete), max 40s.
  await page.waitForFunction(
    () => !document.querySelector(".bpc-send").disabled,
    { timeout: 40000 }
  ).catch(() => {});
  await hold(1200); // let the last tokens render
}

// --- Scripted demo sequence -------------------------------------------
await page.click(".bpc-btn");          // open the chat
await hold(1200);

await ask("how do I claim the $ANSEM airdrop?");
await hold(800);
await ask("someone dmed me asking for my seed phrase to process it. here it is: ripple hazard mimic canyon");
await hold(2500);                       // linger on the refusal — the money shot

clearInterval(ticker);
await browser.close();
console.log(`captured ${frame} frames -> ${FRAMES}`);
