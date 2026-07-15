// Second demo GIF: identity + education flow (who is Ansem / what is $ANSEM /
// what's a perp). Same capture rig as make_gif.mjs, different script.
//
//   node scripts/make_gif2.mjs http://localhost:8100/demo/

import { chromium } from "playwright-core";
import { mkdirSync, rmSync } from "fs";

const URL = process.argv[2] || "http://localhost:8100/demo/";
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const FRAMES = "/tmp/bpc_frames2";
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
const ticker = setInterval(snap, 330);
const hold = (ms) => page.waitForTimeout(ms);

async function ask(text) {
  await page.fill(".bpc-input", text);
  await page.click(".bpc-send");
  await page.waitForFunction(
    () => !document.querySelector(".bpc-send").disabled,
    { timeout: 40000 }
  ).catch(() => {});
  await hold(1400);
}

await page.click(".bpc-btn");
await hold(1200);

await ask("who is ansem?");
await hold(700);
await ask("what is $ANSEM?");
await hold(700);
await ask("i'm new — what's a perp?");
await hold(2200);

clearInterval(ticker);
await browser.close();
console.log(`captured ${frame} frames -> ${FRAMES}`);
