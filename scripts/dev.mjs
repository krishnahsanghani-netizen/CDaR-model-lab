#!/usr/bin/env node
import { spawnSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

const ROOT = process.cwd();
const VENV_DIR = path.join(ROOT, ".venv");
const READY_MARKER = path.join(VENV_DIR, ".cdar_ready");
const MATPLOTLIB_DIR = path.join(ROOT, ".matplotlib");

function run(cmd, args, opts = {}) {
  const result = spawnSync(cmd, args, {
    stdio: "inherit",
    cwd: ROOT,
    env: process.env,
    ...opts,
  });
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

function tryRunCapture(cmd, args) {
  const result = spawnSync(cmd, args, {
    stdio: ["ignore", "pipe", "pipe"],
    cwd: ROOT,
    env: process.env,
    encoding: "utf-8",
  });
  return result;
}

function resolvePython() {
  const candidates = [
    process.env.PYTHON_CMD,
    "python3.11",
    "python3.10",
    "python3",
  ].filter(Boolean);

  for (const candidate of candidates) {
    const version = tryRunCapture(String(candidate), [
      "-c",
      "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
    ]);
    if (version.status !== 0) continue;
    const v = (version.stdout || "").trim();
    const [major, minor] = v.split(".").map((x) => Number(x));
    if (major > 3 || (major === 3 && minor >= 10)) {
      return String(candidate);
    }
  }

  console.error("Python 3.10+ not found. Install Python 3.11 first.");
  process.exit(1);
}

function ensureVenv(pythonCmd) {
  if (!existsSync(VENV_DIR)) {
    console.log("Creating virtual environment...");
    run(pythonCmd, ["-m", "venv", ".venv"]);
  }
}

function ensureDeps(venvPython) {
  if (!existsSync(READY_MARKER)) {
    console.log("Installing dependencies (this may take a minute)...");
    run(venvPython, ["-m", "pip", "install", "-e", ".[dev]"]);
    writeFileSync(READY_MARKER, new Date().toISOString(), "utf-8");
  }
}

function main() {
  const pythonCmd = resolvePython();
  ensureVenv(pythonCmd);

  const venvPython = path.join(VENV_DIR, "bin", "python");
  ensureDeps(venvPython);

  if (!existsSync(MATPLOTLIB_DIR)) {
    mkdirSync(MATPLOTLIB_DIR, { recursive: true });
  }

  const passthrough = process.argv.slice(2);
  const baseArgs = [
    "-m",
    "streamlit",
    "run",
    "ui/streamlit_app.py",
    "--server.port",
    process.env.PORT || "8501",
    "--server.headless",
    "true",
  ];

  const child = spawn(venvPython, [...baseArgs, ...passthrough], {
    stdio: "inherit",
    cwd: ROOT,
    env: {
      ...process.env,
      MPLCONFIGDIR: MATPLOTLIB_DIR,
    },
  });

  child.on("exit", (code) => process.exit(code ?? 0));
}

main();
