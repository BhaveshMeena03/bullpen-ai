// Render any demo/*-card.html to a PNG next to it.
//   node scripts/make_card.mjs three-archives 1600 760
//   node scripts/make_card.mjs how-to-ask 1600 720
//
// One script rather than one per card: these are all the same job, and a
// second copy of it would drift from the first the moment either changed.
// Rendered at 2x because X re-encodes whatever it is handed, so it is
// better to give it pixels to throw away than to ask it to invent any.
import { chromium } from "playwright-core";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const [name, w = "1600", h = "760"] = process.argv.slice(2);
if (!name) {
  console.error("usage: node scripts/make_card.mjs <name> [width] [height]");
  process.exit(1);
}

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const width = Number(w), height = Number(h);

const browser = await chromium.launch({ executablePath: CHROME });
const page = await browser.newPage({
  viewport: { width, height },
  deviceScaleFactor: 2,
});
await page.goto("file://" + join(root, "demo", `${name}-card.html`));
await page.waitForTimeout(500);
const out = join(root, "demo", `${name}-card.png`);
await page.screenshot({ path: out });
await browser.close();
console.log(`wrote demo/${name}-card.png (${width * 2}x${height * 2})`);
