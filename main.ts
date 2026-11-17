// GTA VI launch-day Twitch forecast.
//
// Reads each case in the input directory (schema makefun/twitch-ccu-case/v1):
// hour 1's print, the target's history, and past launches' hourly curves
// (comparables). Writes one forecast per case: for hours 2..24, five
// quantiles (5/25/50/75/95%) of that hour's concurrent-viewer count.
//
// Method (chosen by leave-one-out backtest over the kit's comparables):
//   1. For every comparable launch, take the log-ratio of each hour's
//      viewers to its own hour-1 print. Pool those ratios per hour.
//   2. Weight each comparable by similarity to a rolling-midnight,
//      console-only blockbuster: rolling unlocks, console-only platforms,
//      recent launches, and hour-1 scale close to the target's.
//   3. Take weighted empirical quantiles of the pooled ratios per hour.
//   4. Correct the level with a shrunk per-hour regression of the ratio
//      on log hour-1 size: bigger launches ramp differently from small ones.
//   5. Narrow the band slightly around the median (the raw empirical
//      quantiles run a little wide in backtest).
// No network, no clock, no randomness: the same input always yields the
// same forecast bytes.

import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const MAX_VIEWERS = 1e9;

// Tuned by leave-one-out backtest on the kit's 37 usable comparables.
const WEIGHT_ROLLING = 1.6;
const WEIGHT_SIMULTANEOUS = 0.6;
const WEIGHT_CONSOLE_ONLY = 1.4;
const WEIGHT_RECENT = 1.3;
const RECENT_FROM_MS = Date.parse("2024-01-01T00:00:00Z");
const SCALE_DECAY = 0.65;
const REGRESSION_SHRINK = 0.5;
const SPREAD = 0.9;
// Fallback spread when an hour has no comparable ratios at all.
const FALLBACK_SIGMA = 0.15;
const NORMAL_Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

function hourKey(ms) {
  return new Date(ms).toISOString().slice(0, 13);
}

function tierOf(comp) {
  const m = /Tier: ([a-z-]+)/.exec(comp.unlock || "");
  return m ? m[1] : "unknown";
}

function unlockKindOf(comp) {
  const u = (comp.unlock || "").toLowerCase();
  if (u.includes("rolling") || u.includes("midnight local")) return "rolling";
  if (u.includes("simultaneous") || u.includes("single global")) return "simultaneous";
  return "unknown";
}

function isConsoleOnly(comp) {
  return !(comp.platforms || "").toLowerCase().includes("pc");
}

// Hourly viewers for one comparable, keyed by UTC hour, plus its day curve.
function prepareComparable(comp) {
  const curve = new Map();
  for (const p of comp.hourly || []) {
    curve.set(hourKey(Date.parse(p.from)), Math.max(1, p.viewers));
  }
  const dayMs = Date.parse(comp.day_start);
  const day = [];
  for (let h = 1; h <= 24; h++) {
    day.push(curve.get(hourKey(dayMs + (h - 1) * 3600e3)) ?? null);
  }
  return {
    tier: tierOf(comp),
    kind: unlockKindOf(comp),
    consoleOnly: isConsoleOnly(comp),
    dayMs,
    day,
  };
}

function similarityWeight(comp, logHour1) {
  let w = 1;
  if (comp.kind === "rolling") w *= WEIGHT_ROLLING;
  if (comp.kind === "simultaneous") w *= WEIGHT_SIMULTANEOUS;
  if (comp.consoleOnly) w *= WEIGHT_CONSOLE_ONLY;
  if (comp.dayMs >= RECENT_FROM_MS) w *= WEIGHT_RECENT;
  w *= Math.exp(-SCALE_DECAY * Math.abs(Math.log(comp.day[0]) - logHour1));
  return w;
}

function weightedQuantile(pairs, q) {
  let total = 0;
  for (const [, w] of pairs) total += w;
  if (total <= 0) return 0;
  let acc = 0;
  for (const [v, w] of pairs) {
    acc += w;
    if (acc >= q * total) return v;
  }
  return pairs[pairs.length - 1][0];
}

function forecastCase(kase) {
  // `known` is documented as a one-row array; accept a bare object too.
  const knownRows = Array.isArray(kase.known) ? kase.known : [kase.known];
  const knownRow = knownRows[0] || {};
  let hour1 = knownRow.viewers;
  if (typeof hour1 !== "number" || !Number.isFinite(hour1) || hour1 < 1) {
    const hourly = kase.history_hourly || [];
    hour1 = hourly.length ? hourly[hourly.length - 1].viewers : 1;
  }
  hour1 = Math.max(1, hour1);
  const logHour1 = Math.log(hour1);

  // Pool: past launches only. The target game's own pre-release history
  // (tier target-history, or a comparable named after the target game)
  // is not a launch and never enters the ratio pool.
  const targetName = (kase.game && kase.game.name) || "";
  const pool = [];
  for (const comp of kase.comparables || []) {
    const prepared = prepareComparable(comp);
    if (prepared.tier === "target-history") continue;
    if (targetName && (comp.name || "").includes(targetName)) continue;
    if (prepared.day[0] == null) continue;
    pool.push(prepared);
  }

  const logRatios = {};
  for (let h = 2; h <= 24; h++) {
    const pairs = [];
    for (const comp of pool) {
      if (comp.day[h - 1] == null) continue;
      const ratio = Math.log(comp.day[h - 1]) - Math.log(comp.day[0]);
      pairs.push([ratio, similarityWeight(comp, logHour1)]);
    }
    pairs.sort((a, b) => a[0] - b[0]);
    logRatios[h] = pairs;
  }

  // Weighted empirical quantiles of the log-ratios, per hour.
  const quantiles = {};
  for (let h = 2; h <= 24; h++) {
    const pairs = logRatios[h];
    if (pairs.length === 0) {
      quantiles[h] = NORMAL_Z.map((z) => z * FALLBACK_SIGMA * Math.sqrt(h - 1));
    } else {
      quantiles[h] = LEVELS.map((tau) => weightedQuantile(pairs, tau));
    }
  }

  // Shrunk scale regression: ratios drift with launch size. Fit one
  // weighted line per hour (ratio on log hour-1), then shift the whole
  // band by the shrunk slope times the target's distance from the
  // pool's weighted mean log hour-1.
  for (let h = 2; h <= 24; h++) {
    const pairs = logRatios[h];
    if (pairs.length < 5) continue;
    let sw = 0;
    let sx = 0;
    let sy = 0;
    let sxx = 0;
    let sxy = 0;
    for (const comp of pool) {
      if (comp.day[h - 1] == null) continue;
      const w = similarityWeight(comp, logHour1);
      const x = Math.log(comp.day[0]);
      const y = Math.log(comp.day[h - 1]) - x;
      sw += w;
      sx += w * x;
      sy += w * y;
      sxx += w * x * x;
      sxy += w * x * y;
    }
    const denom = sw * sxx - sx * sx;
    if (denom <= 1e-9) continue;
    const slope = (sw * sxy - sx * sy) / denom;
    const shift = REGRESSION_SHRINK * slope * (logHour1 - sx / sw);
    quantiles[h] = quantiles[h].map((q) => q + shift);
  }

  const hours = [];
  for (let h = 2; h <= 24; h++) {
    const qs = quantiles[h];
    const median = qs[2];
    const row = qs.map((q) => {
      const v = Math.exp(logHour1 + median + (q - median) * SPREAD);
      return Math.min(MAX_VIEWERS, Math.max(1, v));
    });
    // Keep the row non-decreasing after clamping.
    for (let k = 1; k < row.length; k++) {
      if (row[k] < row[k - 1]) row[k] = row[k - 1];
    }
    hours.push({ hour: h, q: row });
  }
  return { hours };
}

function main() {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  mkdirSync(outputDir, { recursive: true });
  const names = readdirSync(inputDir).filter((n) => n.endsWith(".in")).sort();
  for (const name of names) {
    const kase = JSON.parse(readFileSync(join(inputDir, name), "utf8"));
    const out = forecastCase(kase);
    const outName = name.slice(0, -3) + ".out";
    writeFileSync(join(outputDir, outName), JSON.stringify(out));
  }
}

main();
