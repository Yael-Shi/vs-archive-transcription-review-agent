"""Read-only HTML packet for the first human-verification queue.

The packet copies original page images and renders recorded readings.
It does not approve a reading, draw an unrecorded crop, infer a writer,
or append a ledger.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from transcription_review_agent.bundling.hashing import sha256_bytes, sha256_text
from transcription_review_agent.bundling.publish import canonical_json
from transcription_review_agent.constants import SHA256_PATTERN
from transcription_review_agent.review.errors import (
    ReviewError,
    ReviewIntegrityError,
    ReviewPathError,
    ReviewUsageError,
)
from transcription_review_agent.review.localization import (
    PILOT_DISPUTES,
    CaseBinding,
    LocalizationMark,
    LocalizationSidecar,
    NormalizedBox,
    bind_sidecar,
    export_sidecar,
    is_pilot_dispute,
)
from transcription_review_agent.review.visual_localization import (
    ProposalBinding,
    VisualProposal,
    VisualProposalSidecar,
    bind_visual_proposals,
    export_visual_proposals,
)
from transcription_review_agent.review.learning_export import (
    EvidenceRefRecord,
    LearningCandidate,
    LearningCandidateExport,
    PageRef,
    export_learning_candidates,
)
from transcription_review_agent.review.models import (
    DISPUTE_PRIORITY,
    BBox,
    CropRecord,
    DisputeCategory,
)
from transcription_review_agent.review.paths import contained, reviews_root

PACKET_SCHEMA: Final = "human-verification-packet-v1"
PHASE1_DOCUMENT_IDS: tuple[int, ...] = (104, 105, 108)
_LOCATED_SPANS = frozenset({"RECORDED_REGION_BBOX", "RECORDED_TARGETED_WINDOW"})
_UNLOCATED_SPANS = frozenset({"UNAVAILABLE", "CONFLICTING_REGION_BBOX"})
_IMAGE_SUFFIX = {"image/jpeg": ".jpg", "image/png": ".png"}
_PACKET_GENERATED = frozenset({"index.html", "packet_manifest.json"})
_PROTECTED_SIDECARS = frozenset({"localization.json", "visual_proposals.json"})
_PACKET_IMAGE_NAME = re.compile(r"^[0-9a-f]{64}\.(?:jpg|jpeg|png)$")
_PACKET_TEMP = re.compile(
    r"^\.(?:index\.html|packet_manifest\.json|localization\.json|"
    r"visual_proposals\.json|[0-9a-f]{64}\.(?:jpg|jpeg|png))"
    r"\.(?P<pid>[0-9]+)\.[0-9a-f]{12}\.tmp$"
)
_PACKET_NOTES: tuple[str, ...] = (
    "זו תצוגה לקריאה. אין כאן אישור קריאה ואין כאן רישום החלטה.",
    "הסכמה בין שתי הקריאות אינה בדיקה של אדם.",
    "תמונת העמוד המקורית מועתקת פעם אחת לכל SHA-256. קובצי PNG נגזרים אינם מועתקים.",
    "קטע ברמת מילה נכלל רק כשהייצוא רשם חלון ממוקד והגיבוב תואם. נפילה לעמוד מלא אינה קטע.",
    "זהות הכותב לא נרשמה ולא מוסקת.",
    "לא מומצאת קואורדינטה. תיבה מהייצוא מצוירת רק כשנשמר image_bbox אחד.",
    "אזור שמסומן בדפדפן נשאר בטאב עד שמייצאים את קובץ הסימונים.",
    "מסגרת כחולה מקווקו היא הצעת מיקום. היא אינה מאשרת את המילים.",
    "מסגרת ירוקה היא מיקום שאדם אישר בעמוד. גם היא אינה מאשרת את המילים.",
)

_STYLE = """
html, body { height: 100%; }
body { margin: 0; height: 100vh; overflow: hidden; display: flex; flex-direction: column; color: #2c261f; background: #f3eee6; font: 17px/1.45 "Segoe UI", "Arial Hebrew", Arial, sans-serif; }
nav { position: sticky; top: 0; flex: 0 0 auto; z-index: 2; display: flex; flex-wrap: wrap; gap: .45rem .6rem; align-items: center; padding: .45rem .75rem; background: #fffdf8; border-bottom: 1px solid #e4d8c6; }
#prev, #next { font: inherit; font-size: 1.05rem; padding: .45rem 1.05rem; border: 0; border-radius: 999px; background: #2c261f; color: #fffdf8; }
.progress { flex: 1 1 12rem; min-width: 10rem; text-align: center; }
#position { display: block; font-size: 1.05rem; }
#case-meta, #view-status { display: block; color: #6d6458; font-size: .92rem; }
#view-status { min-height: 1.2em; }
.view-controls, .region-jumps, .tools { display: flex; flex-wrap: wrap; gap: .35rem; }
.view-controls button, .region-jumps button, .tools button, details.tools-drawer > summary, .technical > summary { font: inherit; font-size: .92rem; padding: .32rem .7rem; border: 1px solid #d9cbb6; border-radius: 999px; background: #fff; color: #2c261f; }
details.tools-drawer { margin-inline-start: auto; }
details.tools-drawer > div { display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; padding: .45rem 0 .1rem; }
#save-state { flex-basis: 100%; margin: 0; color: #6d6458; font-size: .85rem; }
main { flex: 1 1 auto; min-height: 0; }
article.case { display: none; height: 100%; }
article.case.active { display: block; }
.layout { display: grid; grid-template-columns: minmax(320px, .78fr) minmax(0, 1.28fr); grid-template-areas: "text image"; height: 100%; min-height: 0; }
.viewport { grid-area: image; overflow: auto; height: 100%; min-height: 0; background: #241f1b; touch-action: none; direction: ltr; }
.sheet { position: relative; direction: ltr; margin-inline: auto; }
.sheet img.page { display: block; width: 100%; height: auto; }
.sheet .box, .sheet .mark, .sheet .draft, .sheet .proposal, .sheet .confirmed { position: absolute; box-sizing: border-box; pointer-events: none; }
.sheet .box { border: 2px solid #c4a15a; }
.sheet .mark { border: 2px solid #6d8f45; background: rgba(109, 143, 69, .12); }
.sheet .draft { border: 2px dashed #6d8f45; }
.sheet .proposal { border: 2px dashed #2f5f8a; background: rgba(47, 95, 138, .1); }
.sheet .confirmed { border: 2px solid #2f6f5e; background: rgba(47, 111, 94, .1); }
.sheet .focused { outline: 3px solid #c47b2b; outline-offset: 2px; }
.proposal-caption, .confirmed-caption { display: none; position: absolute; inset-inline-start: 0; bottom: 100%; color: #fff; font: 12px/1.2 "Segoe UI", Arial, sans-serif; padding: 1px 5px; white-space: nowrap; }
.focused > .proposal-caption, .focused > .confirmed-caption { display: block; }
.proposal-caption { background: #2f5f8a; }
.confirmed-caption { background: #2f6f5e; }
.read { grid-area: text; overflow: auto; height: 100%; min-height: 0; padding: .7rem .9rem 1rem; direction: rtl; background: #f7f3ec; }
.kicker { margin: 0; color: #6d6458; }
.read h2 { margin: .1rem 0 .35rem; font-size: 1.35rem; font-weight: 650; }
.disagreement { margin: 0 0 .55rem; }
.disagreement p, .where { margin: .2rem 0; }
.where { color: #3d3832; }
.readings { display: grid; grid-template-columns: 1fr 1fr; gap: .45rem; }
.readings h3 { margin: 0; font-size: .95rem; }
.who { margin: 0 0 .15rem; color: #6d6458; font-size: .82rem; }
pre.reading { white-space: pre-wrap; background: #fff; border: 1px solid #e4d8c6; border-radius: 8px; padding: .45rem .55rem; margin: 0; max-height: 24vh; overflow: auto; unicode-bidi: plaintext; text-align: start; }
p.empty { margin: 0; padding: .45rem .55rem; background: #fff; border: 1px dashed #e4d8c6; border-radius: 8px; color: #6d6458; }
.region-jumps { margin: .45rem 0; }
.technical { margin-top: .6rem; }
.technical > summary { display: inline-block; }
.status { display: inline-block; margin-top: .4rem; color: #6d6458; font-size: .8rem; }
dl { margin: .4rem 0; }
dt { color: #6d6458; font-size: .82rem; }
dd { margin: 0 0 .35rem; direction: ltr; text-align: start; overflow-wrap: anywhere; }
img.crop { max-width: 100%; height: auto; }
.packet-notes { margin: 0; padding: .2rem .8rem .6rem; background: #fffdf8; border-top: 1px solid #e4d8c6; }
@media (max-width: 720px) {
  body { height: auto; overflow: auto; }
  article.case { height: auto; }
  .layout { grid-template-columns: 1fr; grid-template-areas: "image" "text"; }
  .viewport { height: 46vh; }
  .readings { grid-template-columns: 1fr; }
  pre.reading { max-height: none; }
  details.tools-drawer { margin-inline-start: 0; }
}
"""

_SCRIPT = r"""
(function () {
  var SCALE = 1000000;
  var cases = Array.prototype.slice.call(document.querySelectorAll("article.case"));
  var marks = {};
  var dirty = false;
  var active = 0;
  var mode = "pan";
  var zoom = 1;
  var userAdjusted = false;
  var docFilter = document.getElementById("filter-document");
  var categoryFilter = document.getElementById("filter-category");
  var pilotFilter = document.getElementById("filter-pilot");
  var saveState = document.getElementById("save-state");
  var viewStatus = document.getElementById("view-status");

  function setView(text) {
    if (viewStatus) viewStatus.textContent = text;
  }

  function ident(article) { return article.getAttribute("data-candidate-id") || ""; }

  function matches(article) {
    if (docFilter.value !== "all" && article.getAttribute("data-document") !== docFilter.value) {
      return false;
    }
    if (categoryFilter.value !== "all" && article.getAttribute("data-category") !== categoryFilter.value) {
      return false;
    }
    if (pilotFilter.checked && article.getAttribute("data-pilot") !== "1") {
      return false;
    }
    return true;
  }

  function visibleIndexes() {
    var found = [];
    cases.forEach(function (article, index) {
      if (matches(article)) found.push(index);
    });
    return found;
  }

  function applyZoom(article) {
    var sheet = article.querySelector(".sheet");
    var native = Number(sheet.getAttribute("data-native-width"));
    if (!native || native < 1) return;
    sheet.style.width = String(Math.round(native * zoom)) + "px";
  }

  function fitZoom(viewport, nativeW, nativeH) {
    if (!viewport.clientWidth || !viewport.clientHeight || !nativeW || !nativeH) return 1;
    return Math.min(viewport.clientWidth / nativeW, viewport.clientHeight / nativeH) * 0.94;
  }

  function fitPage(article) {
    var viewport = article.querySelector(".viewport");
    var sheet = article.querySelector(".sheet");
    var nativeW = Number(sheet.getAttribute("data-native-width"));
    var nativeH = Number(sheet.getAttribute("data-native-height"));
    zoom = fitZoom(viewport, nativeW, nativeH);
    applyZoom(article);
    viewport.scrollLeft = 0;
    viewport.scrollTop = 0;
  }

  function paint(article) {
    var sheet = article.querySelector(".sheet");
    var existing = sheet.querySelector(".mark.human");
    if (existing) existing.remove();
    var status = article.querySelector(".loc-status");
    var mark = marks[ident(article)];
    if (!mark) {
      status.textContent = "לא סומן מיקום.";
      return;
    }
    if (mark.status === "HUMAN_MARKED" && mark.bbox) {
      status.textContent = "נשמר סימון ידני בטאב הזה. הקריאה לא אומתה.";
      var box = document.createElement("div");
      box.className = "mark human";
      box.setAttribute("data-provenance", "HUMAN_MARKED");
      box.style.left = (mark.bbox.x / SCALE * 100) + "%";
      box.style.top = (mark.bbox.y / SCALE * 100) + "%";
      box.style.width = (mark.bbox.width / SCALE * 100) + "%";
      box.style.height = (mark.bbox.height / SCALE * 100) + "%";
      sheet.appendChild(box);
      return;
    }
    if (mark.status === "UNCLEAR") {
      status.textContent = "סומן שהמיקום לא ברור. לא נשמרה תיבה.";
      return;
    }
    if (mark.status === "NO_LOCATION") {
      status.textContent = "סומן שאין מיקום. לא נשמרה תיבה.";
    }
  }

  function focusBox(article, kind, region) {
    var selector = kind === "confirmed" ? ".confirmed" : ".proposal";
    var boxes = article.querySelectorAll(selector);
    var box = null;
    for (var i = 0; i < boxes.length; i += 1) {
      if (boxes[i].getAttribute("data-region") === String(region)) box = boxes[i];
    }
    if (!box && boxes.length) box = boxes[0];
    article.querySelectorAll(".focused").forEach(function (node) {
      node.classList.remove("focused");
    });
    if (!box) {
      fitPage(article);
      setView("אין אזור מסומן. מוצג העמוד כולו.");
      return;
    }
    var viewport = article.querySelector(".viewport");
    var sheet = article.querySelector(".sheet");
    var nativeW = Number(sheet.getAttribute("data-native-width"));
    var nativeH = Number(sheet.getAttribute("data-native-height"));
    var nw = Number(box.getAttribute("data-nw"));
    var nh = Number(box.getAttribute("data-nh"));
    var nx = Number(box.getAttribute("data-nx"));
    var ny = Number(box.getAttribute("data-ny"));
    var boxW = Math.max(1, (nw / SCALE) * nativeW);
    var boxH = Math.max(1, (nh / SCALE) * nativeH);
    var fit = fitZoom(viewport, nativeW, nativeH);
    var padded = Math.min(
      viewport.clientWidth / (boxW * 2.6),
      viewport.clientHeight / (boxH * 3.2)
    );
    zoom = Math.max(fit, Math.min(1.8, padded));
    if (!isFinite(zoom) || zoom <= 0) zoom = fit || 1;
    userAdjusted = true;
    applyZoom(article);
    var cx = ((nx + nw / 2) / SCALE) * sheet.offsetWidth;
    var cy = ((ny + nh / 2) / SCALE) * sheet.offsetHeight;
    viewport.scrollLeft = Math.max(0, cx - viewport.clientWidth / 2);
    viewport.scrollTop = Math.max(0, cy - viewport.clientHeight / 2);
    box.classList.add("focused");
    setView(kind === "confirmed"
      ? "מוצג המיקום שאושר בעמוד, עם השורות סביבו. הקריאה לא אומתה."
      : "מוצגת הצעת המיקום, עם השורות סביבה. ההצעה לא מאשרת את הקריאה.");
  }

  function show(index) {
    var visible = visibleIndexes();
    if (visible.length === 0) {
      cases.forEach(function (article) { article.classList.remove("active"); });
      document.getElementById("position").textContent = "0 / 0";
      return;
    }
    if (visible.indexOf(index) === -1) index = visible[0];
    active = index;
    cases.forEach(function (article, item) {
      article.classList.toggle("active", item === active);
    });
    var place = visible.indexOf(active);
    document.getElementById("position").textContent = (place + 1) + " מתוך " + visible.length;
    var current = cases[active];
    var meta = document.getElementById("case-meta");
    if (meta) {
      meta.textContent = "מסמך " + current.getAttribute("data-document")
        + " · עמוד " + (current.getAttribute("data-page-label") || "");
    }
    document.getElementById("mode-draw").disabled = current.getAttribute("data-pilot") !== "1";
    if (current.getAttribute("data-pilot") !== "1") mode = "pan";
    userAdjusted = false;
    setView("מוצג העמוד כולו.");
    paint(current);
    function showPage() {
      if (cases[active] !== current || userAdjusted) return;
      fitPage(current);
    }
    var image = current.querySelector("img.page");
    if (image && !image.complete) image.addEventListener("load", showPage, {once: true});
    showPage();
    try { history.replaceState(null, "", "#" + current.id); } catch (error) { location.hash = current.id; }
    function pinTop() {
      var scroller = document.scrollingElement;
      if (scroller) scroller.scrollTop = 0;
    }
    pinTop();
    requestAnimationFrame(pinTop);
  }

  function step(delta) {
    var visible = visibleIndexes();
    if (visible.length === 0) return;
    var place = visible.indexOf(active);
    if (place === -1) place = 0;
    var next = (place + delta + visible.length) % visible.length;
    show(visible[next]);
  }

  function point(sheet, event) {
    var rect = sheet.getBoundingClientRect();
    var x = (event.clientX - rect.left) / rect.width;
    var y = (event.clientY - rect.top) / rect.height;
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > 1) x = 1;
    if (y > 1) y = 1;
    return {x: x, y: y};
  }

  function normalize(box) {
    var x = Math.round(box.x * SCALE);
    var y = Math.round(box.y * SCALE);
    var width = Math.round(box.w * SCALE);
    var height = Math.round(box.h * SCALE);
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x + width > SCALE) width = SCALE - x;
    if (y + height > SCALE) height = SCALE - y;
    if (width < 1 || height < 1) return null;
    if (x === 0 && y === 0 && width === SCALE && height === SCALE) return null;
    return {x: x, y: y, width: width, height: height, scale: SCALE};
  }

  function record(article, status, bbox) {
    return {
      candidate_id: ident(article),
      document_id: Number(article.getAttribute("data-document")),
      page_id: article.getAttribute("data-page-id"),
      page_image_sha256: article.getAttribute("data-page-sha"),
      page_width: Number(article.getAttribute("data-page-width")),
      page_height: Number(article.getAttribute("data-page-height")),
      status: status,
      method: "HUMAN_MARK",
      bbox: bbox,
      coordinate_space: "NORMALIZED_IMAGE",
      human_verification_status: "NOT_VERIFIED",
      eligible_for_authoritative_learning: false
    };
  }

  function setDirty(text) {
    dirty = true;
    saveState.textContent = text;
  }

  function articleFor(id) {
    for (var i = 0; i < cases.length; i += 1) {
      if (ident(cases[i]) === id) return cases[i];
    }
    return null;
  }

  function acceptImported(payload) {
    if (!payload || payload.schema_version !== "dispute-localization-v1") {
      throw new Error("schema");
    }
    if (payload.human_verified_count !== 0 || payload.decision_ledger_written !== false) {
      throw new Error("verification");
    }
    if (!Array.isArray(payload.marks)) throw new Error("marks");
    var next = {};
    payload.marks.forEach(function (mark) {
      if (!mark || next[mark.candidate_id]) throw new Error("duplicate");
      var article = articleFor(mark.candidate_id);
      if (!article || article.getAttribute("data-pilot") !== "1") throw new Error("case");
      if (mark.page_image_sha256 !== article.getAttribute("data-page-sha")) throw new Error("image");
      if (mark.page_id !== article.getAttribute("data-page-id")) throw new Error("page");
      if (Number(mark.document_id) !== Number(article.getAttribute("data-document"))) throw new Error("document");
      if (Number(mark.page_width) !== Number(article.getAttribute("data-page-width"))) throw new Error("width");
      if (Number(mark.page_height) !== Number(article.getAttribute("data-page-height"))) throw new Error("height");
      if (mark.method !== "HUMAN_MARK") throw new Error("method");
      if (mark.human_verification_status !== "NOT_VERIFIED") throw new Error("verified");
      if (mark.eligible_for_authoritative_learning !== false) throw new Error("eligible");
      if (mark.coordinate_space !== "NORMALIZED_IMAGE") throw new Error("space");
      if (mark.status === "HUMAN_MARKED") {
        var box = mark.bbox;
        if (!box || box.scale !== SCALE) throw new Error("box");
        ["x", "y", "width", "height"].forEach(function (key) {
          if (!Number.isInteger(box[key])) throw new Error("box");
        });
        if (box.width < 1 || box.height < 1) throw new Error("box");
        if (box.x < 0 || box.y < 0 || box.x + box.width > SCALE || box.y + box.height > SCALE) throw new Error("box");
        if (box.x === 0 && box.y === 0 && box.width === SCALE && box.height === SCALE) throw new Error("box");
      } else if (mark.status === "UNCLEAR" || mark.status === "NO_LOCATION") {
        if (mark.bbox) throw new Error("box");
      } else {
        throw new Error("status");
      }
      next[mark.candidate_id] = mark;
    });
    marks = next;
    cases.forEach(paint);
    dirty = true;
    saveState.textContent = "הסימונים נטענו רק לטאב הזה. ייצאו אותם כדי לשמור קובץ. סגירת הדף מוחקת את העותק שבטאב.";
  }

  cases.forEach(function (article) {
    var viewport = article.querySelector(".viewport");
    var sheet = article.querySelector(".sheet");
    var pan = null;
    var draw = null;
    var draft = null;
    viewport.addEventListener("pointerdown", function (event) {
      if (event.button !== 0) return;
      if (mode === "draw" && article.getAttribute("data-pilot") === "1") {
        var origin = point(sheet, event);
        draw = {x: origin.x, y: origin.y, pointer: event.pointerId};
        draft = document.createElement("div");
        draft.className = "draft";
        sheet.appendChild(draft);
        viewport.setPointerCapture(event.pointerId);
        event.preventDefault();
        return;
      }
      pan = {x: event.clientX, y: event.clientY, left: viewport.scrollLeft, top: viewport.scrollTop, pointer: event.pointerId};
      viewport.setPointerCapture(event.pointerId);
    });
    viewport.addEventListener("pointermove", function (event) {
      if (draw && event.pointerId === draw.pointer) {
        var current = point(sheet, event);
        var x = Math.min(draw.x, current.x);
        var y = Math.min(draw.y, current.y);
        var w = Math.abs(current.x - draw.x);
        var h = Math.abs(current.y - draw.y);
        draft.style.left = (x * 100) + "%";
        draft.style.top = (y * 100) + "%";
        draft.style.width = (w * 100) + "%";
        draft.style.height = (h * 100) + "%";
        draw.box = {x: x, y: y, w: w, h: h};
      } else if (pan && event.pointerId === pan.pointer) {
        viewport.scrollLeft = pan.left - (event.clientX - pan.x);
        viewport.scrollTop = pan.top - (event.clientY - pan.y);
      }
    });
    viewport.addEventListener("pointerup", function (event) {
      if (!draw || event.pointerId !== draw.pointer) {
        pan = null;
        return;
      }
      var box = normalize(draw.box || {x: 0, y: 0, w: 0, h: 0});
      draw = null;
      if (draft) draft.remove();
      draft = null;
      if (!box) {
        saveState.textContent = "המלבן ריק או מכסה את כל העמוד. לא נשמר מיקום.";
        return;
      }
      article._draft = box;
      saveState.textContent = "המלבן נמצא רק בטאב הזה. בחרו בשמירת האזור, ואז ייצאו את הקובץ.";
    });
    viewport.addEventListener("wheel", function (event) {
      if (article !== cases[active]) return;
      event.preventDefault();
      userAdjusted = true;
      zoom = event.deltaY < 0 ? Math.min(2.4, zoom * 1.1) : Math.max(0.08, zoom / 1.1);
      applyZoom(article);
    }, {passive: false});
    var keep = article.querySelector(".keep-region");
    if (keep) keep.addEventListener("click", function () {
      if (!article._draft) return;
      marks[ident(article)] = record(article, "HUMAN_MARKED", article._draft);
      article._draft = null;
      setDirty("האזור נשמר רק בטאב הזה. ייצאו את הקובץ כדי לשמור אותו. הקריאה לא אומתה.");
      paint(article);
    });
    var unclear = article.querySelector(".mark-unclear");
    if (unclear) unclear.addEventListener("click", function () {
      article._draft = null;
      marks[ident(article)] = record(article, "UNCLEAR", null);
      setDirty("מיקום לא ברור נשמר רק בטאב הזה. ייצאו את הקובץ כדי לשמור אותו.");
      paint(article);
    });
    var none = article.querySelector(".mark-none");
    if (none) none.addEventListener("click", function () {
      article._draft = null;
      marks[ident(article)] = record(article, "NO_LOCATION", null);
      setDirty("הסימון שאין מיקום נשמר רק בטאב הזה. ייצאו את הקובץ כדי לשמור אותו.");
      paint(article);
    });
    article.querySelectorAll(".inspect-region").forEach(function (button) {
      button.addEventListener("click", function () {
        focusBox(article, button.getAttribute("data-kind"), button.getAttribute("data-region"));
      });
    });
  });

  document.getElementById("prev").addEventListener("click", function () { step(-1); });
  document.getElementById("next").addEventListener("click", function () { step(1); });
  [docFilter, categoryFilter, pilotFilter].forEach(function (node) {
    node.addEventListener("change", function () { show(active); });
  });
  document.getElementById("zoom-in").addEventListener("click", function () {
    userAdjusted = true;
    zoom = Math.min(2.4, zoom * 1.25);
    applyZoom(cases[active]);
  });
  document.getElementById("zoom-out").addEventListener("click", function () {
    userAdjusted = true;
    zoom = Math.max(0.08, zoom / 1.25);
    applyZoom(cases[active]);
  });
  document.getElementById("fit-page").addEventListener("click", function () {
    userAdjusted = false;
    fitPage(cases[active]);
    setView("מוצג העמוד כולו.");
  });
  document.getElementById("focus-region").addEventListener("click", function () {
    var current = cases[active];
    if (current.querySelector(".confirmed")) focusBox(current, "confirmed", "0");
    else if (current.querySelector(".proposal")) focusBox(current, "proposal", "0");
    else {
      userAdjusted = false;
      fitPage(current);
      setView("אין אזור מסומן. מוצג העמוד כולו.");
    }
  });
  document.getElementById("mode-pan").addEventListener("click", function () { mode = "pan"; });
  document.getElementById("mode-draw").addEventListener("click", function () {
    if (cases[active].getAttribute("data-pilot") === "1") mode = "draw";
  });
  document.getElementById("export-sidecar").addEventListener("click", function () {
    var payload = {
      schema_version: "dispute-localization-v1",
      packet_role: "MANUAL_LOCALIZATION_SIDECAR",
      human_verified_count: 0,
      decision_ledger_written: false,
      marks: Object.keys(marks).sort().map(function (key) { return marks[key]; })
    };
    var blob = new Blob([JSON.stringify(payload, null, 2) + "\n"], {type: "application/json"});
    var link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "localization.json";
    link.click();
    saveState.textContent = "ההורדה של localization.json התחילה. זה לא אישור שהקובץ נשמר. אם הקובץ לא הופיע, ייצאו שוב. סגירת הטאב מוחקת את הסימונים שבדף.";
  });
  document.getElementById("import-sidecar").addEventListener("change", function (event) {
    var file = event.target.files && event.target.files[0];
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function () {
      try {
        acceptImported(JSON.parse(String(reader.result)));
        show(active);
      } catch (error) {
        saveState.textContent = "הייבוא נדחה. הקובץ לא הוחל. " + error.message;
      }
      event.target.value = "";
    };
    reader.readAsText(file);
  });

  window.addEventListener("beforeunload", function (event) {
    var unsavedDraft = cases.some(function (article) { return !!article._draft; });
    if (!dirty && !unsavedDraft) return;
    event.preventDefault();
    event.returnValue = "";
  });

  var seedNode = document.getElementById("localization-seed");
  var seedText = seedNode ? seedNode.textContent.trim() : "";
  if (seedText && seedText !== "null") {
    try {
      acceptImported(JSON.parse(seedText));
      dirty = false;
      saveState.textContent = "הסימונים שבקובץ נטענו. אחרי שינוי יש לייצא. הקריאה לא אומתה.";
    } catch (error) {
      saveState.textContent = "הסימונים המוטמעים נדחו.";
    }
  }
  var hash = location.hash.replace("#", "");
  var start = cases.findIndex(function (article) { return article.id === hash; });
  show(start === -1 ? 0 : start);
})();
"""


class PacketImage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sha256: str = Field(pattern=SHA256_PATTERN)
    relative_path: str
    mime_type: Literal["image/png", "image/jpeg"]
    role: Literal["ORIGINAL_PAGE", "RECORDED_TARGETED_CROP"]
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class PacketCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    band: Literal[0, 1]
    candidate_id: str = Field(pattern=r"^lc_[0-9a-f]{20}$")
    document_id: int = Field(ge=1)
    batch_id: str
    page_id: str
    page_image_sha256: str = Field(pattern=SHA256_PATTERN)
    image_relative_path: str
    human_verification_status: Literal["NOT_VERIFIED"]
    eligible_for_authoritative_learning: Literal[False]
    learning_category: str
    dispute_category: str
    closure_status: str | None
    image_span_status: str
    image_bbox_recorded: bool


class PacketManifest(BaseModel):
    """Inventory of the HTML packet. It is not a verification ledger."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["human-verification-packet-v1"] = PACKET_SCHEMA
    packet_role: Literal["READ_ONLY_PREVIEW"] = "READ_ONLY_PREVIEW"
    human_verified_count: Literal[0] = 0
    authoritative_learning_eligible_count: Literal[0] = 0
    decision_ledger_written: Literal[False] = False
    document_ids: list[int]
    case_count: int = Field(ge=0)
    band_0_count: int = Field(ge=0)
    band_1_count: int = Field(ge=0)
    export_sha256: str = Field(pattern=SHA256_PATTERN)
    windows_index_path: str | None = None
    limitations: list[str]
    cases: list[PacketCase]
    images: list[PacketImage]


class ConfirmedLocationDisplay(BaseModel):
    """A human-confirmed region drawn for review. The reading stays open."""

    model_config = ConfigDict(extra="ignore")

    candidate_id: str = Field(pattern=r"^lc_[0-9a-f]{20}$")
    dispute_id: str = Field(pattern=r"^dsp_[0-9a-f]{16}$")
    document_id: int = Field(ge=1)
    page_id: str = Field(min_length=1)
    page_image_sha256: str = Field(pattern=SHA256_PATTERN)
    page_width: int = Field(ge=1)
    page_height: int = Field(ge=1)
    boxes: list[NormalizedBox] = Field(min_length=1)
    location_authority: Literal["HUMAN_CONFIRMED_LOCATION"]
    confirmation: Literal["USER_VISUAL_CONFIRMATION"]
    human_verification_status: Literal["NOT_VERIFIED"]
    transcription_verified: Literal[False]


def load_confirmed_location_displays(payload: str) -> list[ConfirmedLocationDisplay]:
    """Read confirmed regions. A verified transcription in that file is refused."""

    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ReviewIntegrityError("confirmed locations are not JSON") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("locations"), list):
        raise ReviewIntegrityError("confirmed locations are missing")
    rows = raw["locations"]
    if not rows:
        raise ReviewIntegrityError("confirmed locations are missing")
    loaded: list[ConfirmedLocationDisplay] = []
    try:
        for row in rows:
            loaded.append(ConfirmedLocationDisplay.model_validate(row))
    except ValidationError as exc:
        raise ReviewIntegrityError(
            "confirmed location does not match the display contract"
        ) from exc
    return loaded


def bind_confirmed_displays(
    rows: Sequence[ConfirmedLocationDisplay], cases: Sequence[QueuedCase]
) -> dict[str, ConfirmedLocationDisplay]:
    """Keep confirmed boxes only when they match a queued pilot case."""

    index = {item.candidate.candidate_id: item for item in cases}
    bound: dict[str, ConfirmedLocationDisplay] = {}
    for row in rows:
        item = index.get(row.candidate_id)
        if item is None:
            continue
        candidate = item.candidate
        if not is_pilot_dispute(
            candidate.document_id, candidate.page_id, candidate.dispute_category
        ):
            raise ReviewIntegrityError(
                "a confirmed location was attached outside the three pilot disputes"
            )
        if (
            row.document_id != candidate.document_id
            or row.page_id != candidate.page_id
            or row.page_image_sha256 != candidate.page_image_sha256
            or row.page_width != item.page.width
            or row.page_height != item.page.height
            or row.dispute_id != candidate.dispute_id
        ):
            raise ReviewIntegrityError(
                "confirmed location does not match the queued case"
            )
        if row.candidate_id in bound:
            raise ReviewIntegrityError("confirmed locations repeat a candidate")
        bound[row.candidate_id] = row
    return bound


@dataclass(frozen=True)
class QueuedCase:
    band: Literal[0, 1]
    candidate: LearningCandidate
    page: PageRef
    superseded: tuple[LearningCandidate, ...]


@dataclass(frozen=True)
class LocatedCrop:
    evidence: EvidenceRefRecord
    path: Path
    width: int
    height: int


def phase1_band(candidate: LearningCandidate) -> Literal[0, 1] | None:
    """Return band 0 or 1, or None when the row is outside the first queue.

    A challenged unresolved dispute stays in band 0 even if a model flag is
    also set. Superseded and partial rows are not queue items.
    """

    if candidate.reading_role != "FULL_DISPUTE":
        return None
    if candidate.human_verification_status != "NOT_VERIFIED":
        return None
    if candidate.eligible_for_authoritative_learning:
        return None
    if candidate.closure_status == "SELECTED_UNRESOLVED":
        return 0
    if candidate.learning_category == "MODEL_SUPPORTED_READING":
        return 1
    return None


def select_phase1_queue(
    export: LearningCandidateExport,
    *,
    document_ids: Sequence[int] = PHASE1_DOCUMENT_IDS,
) -> list[QueuedCase]:
    """Select and rank the first-session queue. The length is not padded."""

    allowed = _unique_ids(document_ids)
    pages = _page_index(export)
    attached = _superseded_by_primary(export)
    selected: list[QueuedCase] = []
    for candidate in export.candidates:
        if candidate.document_id not in allowed:
            continue
        band = phase1_band(candidate)
        if band is None:
            continue
        page = pages.get((candidate.batch_id, candidate.document_id, candidate.page_id))
        if page is None:
            raise ReviewIntegrityError(
                f"candidate {candidate.candidate_id} names a page that is not in the export"
            )
        if (
            page.sha256 != candidate.page_image_sha256
            or page.filename != candidate.page_image_filename
            or page.mime_type != candidate.page_mime_type
        ):
            raise ReviewIntegrityError(
                f"candidate {candidate.candidate_id} does not match its page record"
            )
        _require_box_matches_status(candidate)
        selected.append(
            QueuedCase(
                band=band,
                candidate=candidate,
                page=page,
                superseded=attached.get(candidate.candidate_id, ()),
            )
        )
    selected.sort(key=_sort_key)
    seen: set[str] = set()
    for item in selected:
        if item.candidate.candidate_id in seen:
            raise ReviewIntegrityError("phase-1 queue repeats a candidate id")
        seen.add(item.candidate.candidate_id)
    return selected


def build_human_verification_packet(
    review_dirs: Sequence[Path],
    *,
    output_dir: Path,
    batches_dir: Path | None = None,
    document_ids: Sequence[int] | None = None,
    dry_run: bool = False,
    windows_index_path: str | None = None,
    export: LearningCandidateExport | None = None,
    localization: LocalizationSidecar | None = None,
    visual_proposals: VisualProposalSidecar | None = None,
    confirmed_locations: Sequence[ConfirmedLocationDisplay] | None = None,
) -> PacketManifest:
    """Write the HTML packet, or return its manifest without writing.

    Review directories and sealed bundles are only read. ``dry_run`` reads
    page bytes and writes nothing.
    """

    if not review_dirs:
        raise ReviewUsageError("at least one reviews directory is required")
    roots = [reviews_root(path) for path in review_dirs]
    sealed = reviews_root(batches_dir) if batches_dir is not None else None
    requested = PHASE1_DOCUMENT_IDS if document_ids is None else tuple(document_ids)
    if export is None:
        export = export_learning_candidates(
            roots,
            batches_dir=sealed,
            document_ids=requested,
            dry_run=True,
        )
    cases = select_phase1_queue(export, document_ids=requested)
    if not cases:
        raise ReviewUsageError("phase-1 queue is empty")
    page_files = {
        item.candidate.page_image_sha256: locate_original_page(roots, item.candidate)
        for item in cases
    }
    crop_files: dict[str, LocatedCrop] = {}
    for item in cases:
        for located in locate_targeted_crops(roots, item.candidate):
            digest = located.evidence.crop_sha256
            if digest is None:
                raise ReviewIntegrityError("a targeted crop is missing its hash")
            previous = crop_files.get(digest)
            if (
                previous is not None
                and previous.path.read_bytes() != located.path.read_bytes()
            ):
                raise ReviewIntegrityError("targeted crop hashes collided")
            crop_files[digest] = located
    image_paths: dict[str, str] = {}
    page_meta: dict[str, tuple[str, int, int]] = {}
    for item in cases:
        digest = item.candidate.page_image_sha256
        image_paths[digest] = _image_relative(digest, item.page.mime_type)
        page_meta[digest] = (item.page.mime_type, item.page.width, item.page.height)
    crop_paths = {digest: _image_relative(digest, "image/png") for digest in crop_files}
    limitations = list(export.limitations) + list(_PACKET_NOTES)
    manifest = _manifest(
        cases,
        export=export,
        limitations=limitations,
        windows_index_path=windows_index_path,
        image_paths=image_paths,
        page_meta=page_meta,
        crop_paths=crop_paths,
        crop_sizes={
            digest: (item.width, item.height) for digest, item in crop_files.items()
        },
        document_ids=requested,
    )
    if localization is not None:
        localization = bind_sidecar_to_cases(localization, cases)
    if visual_proposals is not None:
        visual_proposals = bind_proposals_to_cases(
            visual_proposals, cases, export_sha256=manifest.export_sha256
        )
    confirmed = bind_confirmed_displays(confirmed_locations or [], cases)
    if dry_run:
        return manifest
    _refuse_protected(output_dir, roots, sealed)
    page_html = render_packet_html(
        cases,
        limitations=limitations,
        export_sha256=manifest.export_sha256,
        windows_index_path=windows_index_path,
        image_paths=image_paths,
        crop_paths=crop_paths,
        localization=localization,
        visual_proposals=visual_proposals,
        confirmed_locations=confirmed,
    )
    _publish(
        output_dir,
        page_html,
        manifest,
        page_files,
        image_paths,
        crop_files,
        crop_paths,
        localization=localization,
        visual_proposals=visual_proposals,
    )
    return manifest


def render_packet_html(
    cases: Sequence[QueuedCase],
    *,
    limitations: Sequence[str],
    export_sha256: str,
    windows_index_path: str | None,
    image_paths: Mapping[str, str],
    crop_paths: Mapping[str, str],
    localization: LocalizationSidecar | None = None,
    visual_proposals: VisualProposalSidecar | None = None,
    confirmed_locations: Mapping[str, ConfirmedLocationDisplay] | None = None,
) -> str:
    """Render one case at a time. Every inserted value is escaped."""

    if not cases:
        raise ReviewUsageError("phase-1 queue is empty")
    _require_unique_pilots(cases)
    if localization is not None:
        localization = bind_sidecar_to_cases(localization, cases)
    if visual_proposals is not None:
        visual_proposals = bind_proposals_to_cases(
            visual_proposals, cases, export_sha256=export_sha256
        )
    marks = (
        {}
        if localization is None
        else {item.candidate_id: item for item in localization.marks}
    )
    proposals = (
        {}
        if visual_proposals is None
        else {item.candidate_id: item for item in visual_proposals.results}
    )
    confirmed = bind_confirmed_displays(
        [] if confirmed_locations is None else list(confirmed_locations.values()),
        cases,
    )
    documents = _unique_ids(item.candidate.document_id for item in cases)
    categories = _unique_text(item.candidate.dispute_category for item in cases)
    seed = "null" if localization is None else _json_for_html(localization)
    lines = [
        "<!DOCTYPE html>",
        '<html lang="he" dir="rtl">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>בדיקת תמלול</title>",
        "<style>",
        _STYLE,
        "</style>",
        "</head>",
        "<body>",
        "<nav>",
        '<button type="button" id="prev">הקודם</button>',
        '<div class="progress">',
        '<span id="position">1 מתוך 1</span>',
        '<span id="case-meta"></span>',
        '<span id="view-status"></span>',
        "</div>",
        '<button type="button" id="next">הבא</button>',
        '<div class="view-controls">',
        '<button type="button" id="fit-page">עמוד מלא</button>',
        '<button type="button" id="focus-region">מיקוד על המחלוקת</button>',
        '<button type="button" id="zoom-out">התרחקות</button>',
        '<button type="button" id="zoom-in">התקרבות</button>',
        "</div>",
        '<details class="tools-drawer">',
        "<summary>כלים נוספים</summary>",
        "<div>",
        '<button type="button" id="mode-pan">גלילה</button>',
        '<button type="button" id="mode-draw">סימון אזור</button>',
        '<label>מסמך <select id="filter-document"><option value="all">הכל</option>',
        *[
            f'<option value="{document_id}">{document_id}</option>'
            for document_id in documents
        ],
        "</select></label>",
        '<label>קטגוריה <select id="filter-category"><option value="all">הכל</option>',
        *[
            f'<option value="{_esc(category)}">{_esc(_category_label(category))}</option>'
            for category in categories
        ],
        "</select></label>",
        '<label><input id="filter-pilot" type="checkbox"> רק מחלוקות הפיילוט</label>',
        '<button type="button" id="export-sidecar">ייצוא סימונים</button>',
        '<label>ייבוא סימונים <input id="import-sidecar" type="file" accept="application/json,.json"></label>',
        '<p id="save-state">לא נטען קובץ סימונים. אזור שמסמנים נשאר בטאב עד לייצוא.</p>',
        "</div>",
        "</details>",
        "</nav>",
        "<main>",
    ]
    for rank, item in enumerate(cases, start=1):
        lines.append(
            _case_html(
                rank,
                len(cases),
                item,
                image_paths=image_paths,
                crop_paths=crop_paths,
                mark=marks.get(item.candidate.candidate_id),
                proposal=proposals.get(item.candidate.candidate_id),
                confirmed=confirmed.get(item.candidate.candidate_id),
                active=rank == 1,
            )
        )
    lines.extend(
        [
            "</main>",
            '<details class="packet-notes">',
            "<summary>הערות על החבילה</summary>",
            "<ul>",
            *[f"<li>{_esc(item)}</li>" for item in limitations],
            "</ul>",
            f"<p>Export SHA-256: <code>{_esc(export_sha256)}</code></p>",
        ]
    )
    if windows_index_path:
        lines.append(f"<p>Windows path: <code>{_esc(windows_index_path)}</code></p>")
    lines.extend(
        [
            "</details>",
            f'<script type="application/json" id="localization-seed">{seed}</script>',
            "<script>",
            _SCRIPT,
            "</script>",
            "</body>",
            "</html>",
            "",
        ]
    )
    return "\n".join(lines)


def locate_original_page(
    review_dirs: Sequence[Path], candidate: LearningCandidate
) -> Path:
    """Return the original page file whose bytes match the candidate hash."""

    filename = _one_segment(candidate.page_image_filename)
    found: list[Path] = []
    for reviews in review_dirs:
        root = reviews_root(reviews)
        document = (
            root
            / f"batch_{candidate.batch_id}"
            / "documents"
            / f"document_{candidate.document_id}"
        )
        path = contained(document, document / "input" / filename)
        if not path.is_file():
            continue
        if sha256_bytes(path.read_bytes()) != candidate.page_image_sha256:
            raise ReviewIntegrityError(
                f"page image hash does not match candidate {candidate.candidate_id}"
            )
        found.append(path)
    if not found:
        raise ReviewIntegrityError(
            f"missing page image {filename} for document {candidate.document_id}"
        )
    first = found[0].read_bytes()
    for other in found[1:]:
        if other.read_bytes() != first:
            raise ReviewIntegrityError(
                f"page image copies disagree for document {candidate.document_id}"
            )
    return found[0]


def locate_targeted_crops(
    review_dirs: Sequence[Path], candidate: LearningCandidate
) -> list[LocatedCrop]:
    """Resolve recorded targeted crops. Full-page fallbacks are not images."""

    located: list[LocatedCrop] = []
    for evidence in candidate.evidence:
        if evidence.window_kind != "TARGETED":
            continue
        if evidence.crop_sha256 is None:
            raise ReviewIntegrityError(
                f"targeted evidence on {candidate.candidate_id} has no crop hash"
            )
        located.append(_find_targeted_crop(review_dirs, candidate, evidence))
    return located


def windows_path_for(path: Path) -> str | None:
    """Convert a WSL path for Explorer. Image links stay relative either way."""

    version = Path("/proc/version")
    if not version.is_file():
        return None
    text = version.read_text(encoding="utf-8", errors="replace").lower()
    if "microsoft" not in text:
        return None
    try:
        completed = subprocess.run(
            ["wslpath", "-w", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    converted = completed.stdout.strip()
    if not converted or any(mark in converted for mark in "\n\r"):
        return None
    return converted


def main(argv: list[str] | None = None) -> int:
    """Build the packet. Reviews and sealed bundles are not modified."""

    parser = argparse.ArgumentParser(prog="human-verification-packet", add_help=False)
    parser.add_argument("--reviews-dir", action="append", required=True)
    parser.add_argument("--batches-dir")
    parser.add_argument("--document-id", action="append", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--localization")
    parser.add_argument("--visual-proposals")
    parser.add_argument("--confirmed-locations")
    parser.add_argument("--dry-run", action="store_true")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        raise ReviewUsageError(
            "human-verification-packet usage was not recognized. Nothing was written."
        ) from exc
    output_dir = Path(args.output_dir)
    windows = None if args.dry_run else windows_path_for(output_dir / "index.html")
    localization = None
    if args.localization:
        try:
            localization = LocalizationSidecar.model_validate_json(
                Path(args.localization).read_text(encoding="utf-8")
            )
        except ValidationError as exc:
            raise ReviewIntegrityError(
                "localization sidecar does not match the schema"
            ) from exc
    visual_proposals = None
    confirmed_locations = None
    if args.confirmed_locations:
        confirmed_path = Path(args.confirmed_locations)
        _refuse_sidecar_path(
            confirmed_path,
            [Path(path) for path in args.reviews_dir],
            Path(args.batches_dir) if args.batches_dir else None,
        )
        confirmed_locations = load_confirmed_location_displays(
            confirmed_path.read_text(encoding="utf-8")
        )
    if args.visual_proposals:
        proposal_path = Path(args.visual_proposals)
        _refuse_sidecar_path(
            proposal_path,
            [Path(path) for path in args.reviews_dir],
            Path(args.batches_dir) if args.batches_dir else None,
        )
        try:
            visual_proposals = VisualProposalSidecar.model_validate_json(
                proposal_path.read_text(encoding="utf-8")
            )
        except ValidationError as exc:
            raise ReviewIntegrityError(
                "visual proposal sidecar does not match the schema"
            ) from exc
    manifest = build_human_verification_packet(
        [Path(path) for path in args.reviews_dir],
        output_dir=output_dir,
        batches_dir=Path(args.batches_dir) if args.batches_dir else None,
        document_ids=args.document_id,
        dry_run=args.dry_run,
        windows_index_path=windows,
        localization=localization,
        visual_proposals=visual_proposals,
        confirmed_locations=confirmed_locations,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "band_0_count": manifest.band_0_count,
                    "band_1_count": manifest.band_1_count,
                    "case_count": manifest.case_count,
                    "decision_ledger_written": manifest.decision_ledger_written,
                    "dry_run": True,
                    "human_verified_count": manifest.human_verified_count,
                    "packet_written": False,
                },
                sort_keys=True,
            )
        )
        return 0
    print(str(output_dir))
    return 0


def _manifest(
    cases: Sequence[QueuedCase],
    *,
    export: LearningCandidateExport,
    limitations: list[str],
    windows_index_path: str | None,
    image_paths: Mapping[str, str],
    page_meta: Mapping[str, tuple[str, int, int]],
    crop_paths: Mapping[str, str],
    crop_sizes: Mapping[str, tuple[int, int]],
    document_ids: Sequence[int],
) -> PacketManifest:
    images: list[PacketImage] = []
    for digest in sorted(image_paths):
        mime, width, height = page_meta[digest]
        images.append(
            PacketImage(
                sha256=digest,
                relative_path=image_paths[digest],
                mime_type=mime,  # type: ignore[arg-type]
                role="ORIGINAL_PAGE",
                width=width,
                height=height,
            )
        )
    for digest in sorted(crop_paths):
        width, height = crop_sizes[digest]
        images.append(
            PacketImage(
                sha256=digest,
                relative_path=crop_paths[digest],
                mime_type="image/png",
                role="RECORDED_TARGETED_CROP",
                width=width,
                height=height,
            )
        )
    packet_cases = [
        PacketCase(
            rank=rank,
            band=item.band,
            candidate_id=item.candidate.candidate_id,
            document_id=item.candidate.document_id,
            batch_id=item.candidate.batch_id,
            page_id=item.candidate.page_id,
            page_image_sha256=item.candidate.page_image_sha256,
            image_relative_path=image_paths[item.candidate.page_image_sha256],
            human_verification_status="NOT_VERIFIED",
            eligible_for_authoritative_learning=False,
            learning_category=item.candidate.learning_category,
            dispute_category=item.candidate.dispute_category,
            closure_status=item.candidate.closure_status,
            image_span_status=item.candidate.image_span_status,
            image_bbox_recorded=item.candidate.image_bbox is not None,
        )
        for rank, item in enumerate(cases, start=1)
    ]
    return PacketManifest(
        document_ids=list(document_ids),
        case_count=len(packet_cases),
        band_0_count=sum(item.band == 0 for item in cases),
        band_1_count=sum(item.band == 1 for item in cases),
        export_sha256=sha256_text(canonical_json(export)),
        windows_index_path=windows_index_path,
        limitations=limitations,
        cases=packet_cases,
        images=images,
    )


def _publish(
    output_dir: Path,
    page_html: str,
    manifest: PacketManifest,
    page_files: Mapping[str, Path],
    image_paths: Mapping[str, str],
    crop_files: Mapping[str, LocatedCrop],
    crop_paths: Mapping[str, str],
    localization: LocalizationSidecar | None = None,
    visual_proposals: VisualProposalSidecar | None = None,
) -> None:
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".human-verification-", dir=parent))
    try:
        _write_bytes(staging / "index.html", page_html.encode("utf-8"))
        _write_bytes(
            staging / "packet_manifest.json", canonical_json(manifest).encode("utf-8")
        )
        for digest, source in page_files.items():
            _copy_hashed(source, staging / image_paths[digest], digest)
        for digest, located in crop_files.items():
            _copy_hashed(located.path, staging / crop_paths[digest], digest)
        if localization is not None:
            _write_bytes(
                staging / "localization.json",
                export_sidecar(localization).encode("utf-8"),
            )
        if visual_proposals is not None:
            _write_bytes(
                staging / "visual_proposals.json",
                export_visual_proposals(visual_proposals).encode("utf-8"),
            )
        _replace_directory(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _replace_directory(staging: Path, output_dir: Path) -> None:
    """Publish a packet without deleting unrelated files or manual sidecars.

    A missing or empty directory becomes the staging tree. An existing packet
    is updated in place. Anything that is not a packet file stops the publish,
    and a failed update restores the generated files from a private snapshot.
    """

    if output_dir.is_symlink():
        raise ReviewPathError("refusing to replace a packet path that is a link")
    if not output_dir.exists():
        os.replace(staging, output_dir)
        return
    if not output_dir.is_dir():
        raise ReviewPathError(
            "refusing to replace a packet output that is not a directory"
        )
    if any(output_dir.iterdir()):
        stale = _require_known_packet(output_dir, staging)
        _discard_abandoned_temps(stale)
        backup = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}-rollback-", dir=output_dir.parent
            )
        )
        restore_failed = False
        try:
            _snapshot_generated(output_dir, backup)
            try:
                _install_generated(staging, output_dir)
            except Exception as install_error:
                try:
                    _restore_snapshot(backup, output_dir)
                except Exception as restore_error:
                    restore_failed = True
                    raise ReviewIntegrityError(
                        "packet update failed and restore failed; "
                        f"recovery backup kept at {backup}"
                    ) from restore_error
                raise install_error
        finally:
            if not restore_failed:
                shutil.rmtree(backup, ignore_errors=True)
        shutil.rmtree(staging)
        return
    try:
        output_dir.rmdir()
        os.replace(staging, output_dir)
    except Exception:
        if not output_dir.exists():
            output_dir.mkdir(exist_ok=True)
        raise


def _require_known_packet(output_dir: Path, staging: Path) -> list[Path]:
    """Return proven-stale temps. This function does not delete or replace anything."""

    if (
        not (output_dir / "index.html").is_file()
        or not (output_dir / "packet_manifest.json").is_file()
    ):
        raise ReviewIntegrityError(
            "refusing to replace a directory that is not a verification packet"
        )
    stale: list[Path] = []
    for path in output_dir.rglob("*"):
        if path.is_symlink():
            raise ReviewIntegrityError(
                "refusing to replace a packet that contains a link"
            )
        relative = path.relative_to(output_dir).as_posix()
        if path.is_dir():
            if relative != "images":
                raise ReviewIntegrityError(
                    "refusing to replace a packet with an unexpected directory"
                )
            continue
        if _known_packet_file(relative):
            continue
        if _stale_publisher_temp(path):
            stale.append(path)
            continue
        if _PACKET_TEMP.fullmatch(path.name):
            raise ReviewIntegrityError(
                "refusing to replace a packet while a publication temp may still be active"
            )
        raise ReviewIntegrityError(
            "refusing to replace a packet that contains an unexpected file"
        )
    for name in _PROTECTED_SIDECARS:
        existing = output_dir / name
        incoming = staging / name
        if (
            existing.is_file()
            and incoming.is_file()
            and existing.read_bytes() != incoming.read_bytes()
        ):
            raise ReviewIntegrityError(
                "refusing to replace a manually exported sidecar"
            )
    return stale


def _known_packet_file(relative: str) -> bool:
    if relative in _PACKET_GENERATED or relative in _PROTECTED_SIDECARS:
        return True
    prefix, separator, name = relative.partition("/")
    return (
        separator == "/"
        and prefix == "images"
        and "/" not in name
        and _PACKET_IMAGE_NAME.fullmatch(name) is not None
    )


def _snapshot_generated(output_dir: Path, backup: Path) -> None:
    for name in _PACKET_GENERATED:
        source = output_dir / name
        if source.is_file():
            _copy_file(source, backup / name)
    images = output_dir / "images"
    if not images.is_dir():
        return
    for path in images.iterdir():
        if path.is_file() and _PACKET_IMAGE_NAME.fullmatch(path.name):
            _copy_file(path, backup / "images" / path.name)


def _install_generated(staging: Path, output_dir: Path) -> None:
    for name in _PACKET_GENERATED:
        _replace_file(staging / name, output_dir / name)
    staged_images = staging / "images"
    image_root = output_dir / "images"
    incoming = (
        {path.name for path in staged_images.iterdir() if path.is_file()}
        if staged_images.is_dir()
        else set()
    )
    if incoming:
        image_root.mkdir(exist_ok=True)
    for name in sorted(incoming):
        _replace_file(staged_images / name, image_root / name)
    if image_root.is_dir():
        for path in list(image_root.iterdir()):
            if path.is_file() and path.name not in incoming:
                if _PACKET_IMAGE_NAME.fullmatch(path.name):
                    path.unlink()
    for name in _PROTECTED_SIDECARS:
        incoming_side = staging / name
        destination = output_dir / name
        if incoming_side.is_file() and not destination.exists():
            _replace_file(incoming_side, destination)


def _restore_snapshot(backup: Path, output_dir: Path) -> None:
    for name in _PACKET_GENERATED:
        source = backup / name
        destination = output_dir / name
        if source.is_file():
            _replace_file(source, destination)
        elif destination.is_file() and not destination.is_symlink():
            destination.unlink()
    images = output_dir / "images"
    backed = backup / "images"
    keep: set[str] = set()
    if backed.is_dir():
        images.mkdir(exist_ok=True)
        for path in backed.iterdir():
            if path.is_file():
                _replace_file(path, images / path.name)
                keep.add(path.name)
    if images.is_dir():
        for path in list(images.iterdir()):
            if (
                path.is_file()
                and path.name not in keep
                and _PACKET_IMAGE_NAME.fullmatch(path.name)
            ):
                path.unlink()


def _copy_file(source: Path, destination: Path) -> None:
    _write_bytes(destination, source.read_bytes())


def _stale_publisher_temp(path: Path) -> bool:
    """True only for a regular temp file whose recorded publisher process is gone.

    The current process and any pid that still exists, or that cannot be
    checked, are treated as active. Those files are not stale.
    """

    if path.is_symlink() or not path.is_file():
        return False
    match = _PACKET_TEMP.fullmatch(path.name)
    if match is None:
        return False
    pid = int(match.group("pid"))
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def _discard_abandoned_temps(paths: Sequence[Path]) -> None:
    """Unlink temps already classified as abandoned. Recheck before deleting."""

    for path in paths:
        if not _stale_publisher_temp(path):
            raise ReviewIntegrityError(
                "refusing to remove a publication temp whose owner is uncertain"
            )
    for path in paths:
        if not _stale_publisher_temp(path):
            raise ReviewIntegrityError(
                "refusing to remove a publication temp whose owner is uncertain"
            )
        path.unlink()


def _replace_file(source: Path, destination: Path) -> None:
    data = source.read_bytes()
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{sha256_bytes(data)[:12]}.tmp"
    )
    _write_bytes(temporary, data)
    try:
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _copy_hashed(source: Path, destination: Path, expected: str) -> None:
    data = source.read_bytes()
    if sha256_bytes(data) != expected:
        raise ReviewIntegrityError("image hash changed before the packet was written")
    _write_bytes(destination, data)
    if sha256_bytes(destination.read_bytes()) != expected:
        raise ReviewIntegrityError("copied image hash does not match the source")


def _write_bytes(path: Path, data: bytes) -> None:
    """Create ``path`` with every byte. ``os.write`` may return early."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        view = memoryview(data)
        while len(view):
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("incomplete packet write")
            view = view[written:]
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    os.close(descriptor)


def _case_html(
    rank: int,
    total: int,
    item: QueuedCase,
    *,
    image_paths: Mapping[str, str],
    crop_paths: Mapping[str, str],
    mark: LocalizationMark | None,
    active: bool,
    proposal: VisualProposal | None = None,
    confirmed: ConfirmedLocationDisplay | None = None,
) -> str:
    candidate = item.candidate
    page = item.page
    relative = image_paths[candidate.page_image_sha256]
    pilot = is_pilot_dispute(
        candidate.document_id, candidate.page_id, candidate.dispute_category
    )
    active_class = "case active" if active else "case"
    page_label = _page_label(candidate.page_id)
    manual = (
        [
            '<div class="tools">',
            '<button type="button" class="keep-region">שמירת האזור</button>',
            '<button type="button" class="mark-unclear">מיקום לא ברור</button>',
            '<button type="button" class="mark-none">אין מיקום</button>',
            "</div>",
            "<p>סימון אזור זמין רק כאן, בשלוש המחלוקות שנבחרו לניסיון. "
            "הסימון נשאר בטאב עד לייצוא ואינו מאשר את הקריאה.</p>",
        ]
        if pilot
        else [
            "<p>אפשר לעיין במקרה. סימון מיקום מוגבל לשלוש מחלוקות הפיילוט. "
            "לא הומצא כאן אזור.</p>",
        ]
    )
    parts = [
        f'<article class="{active_class}" '
        f'id="case-{_esc(candidate.candidate_id)}" '
        f'data-document="{candidate.document_id}" '
        f'data-page-id="{_esc(candidate.page_id)}" '
        f'data-page-sha="{_esc(candidate.page_image_sha256)}" '
        f'data-page-width="{page.width}" '
        f'data-page-height="{page.height}" '
        f'data-pilot="{1 if pilot else 0}" '
        f'data-band="{item.band}" '
        f'data-page-label="{_esc(page_label)}" '
        f'data-category="{_esc(candidate.dispute_category)}" '
        f'data-candidate-id="{_esc(candidate.candidate_id)}">',
        '<div class="layout">',
        '<div class="read">',
        f'<p class="kicker">מקרה {rank} מתוך {total} · מסמך {candidate.document_id} · '
        f"עמוד {_esc(page_label)}</p>",
        "<h2>מה בודקים?</h2>",
        _disagreement_html(candidate),
        f'<p class="where">{_esc(_where_sentence(proposal, confirmed))}</p>',
        '<div class="readings">',
        "<div>",
        "<h3>GPT</h3>",
        _reading(candidate.gpt_text),
        "</div>",
        "<div>",
        "<h3>Grok</h3>",
        _reading(candidate.grok_text),
        "</div>",
        "</div>",
        _optional_reading("קריאה שנבחרה על ידי מודל", candidate.model_selected_reading),
        '<div class="region-jumps">',
        *_inspect_buttons(proposal, confirmed),
        "</div>",
        '<details class="technical">',
        "<summary>כלים נוספים ופרטים טכניים</summary>",
        f'<p class="loc-status">{_esc(_location_sentence(mark))}</p>',
        *(
            [f'<p class="proposal-status">{_esc(_proposal_sentence(proposal))}</p>']
            if pilot
            else []
        ),
        '<span class="status">NOT_VERIFIED</span>',
        *manual,
        "<dl>",
        _row("מזהה מועמד", candidate.candidate_id),
        _row("אצווה", candidate.batch_id),
        _row("קובץ עמוד", page.filename),
        _row("פיקסלים", f"{page.width}×{page.height}"),
        _row("SHA-256 של העמוד", candidate.page_image_sha256),
        _row("SHA-256 של PNG נגזר", page.derived_png_sha256 or "לא נרשם"),
        _row("מזהה מחלוקת", candidate.dispute_id),
        _row("קטגוריית למידה", candidate.learning_category),
        _row("סטטוס יישוב", candidate.reconciled_status),
        _row("סגירה", candidate.closure_status or "לא נרשם"),
        _row("אימות אנושי", candidate.human_verification_status),
        _row(
            "מסגרת שנרשמה",
            f"{candidate.image_span_status}. {candidate.image_span_limitation}",
        ),
        _row("SHA-256 של החבילה", candidate.source_bundle_sha256),
        _row("SHA-256 של דוח הביקורת", candidate.review_report_sha256),
        _row("SHA-256 של ההשוואה", candidate.comparison_sha256 or "לא נרשם"),
        _row("SHA-256 של פלט GPT", candidate.gpt_output_sha256 or "לא נרשם"),
        _row("SHA-256 של פלט Grok", candidate.grok_output_sha256 or "לא נרשם"),
        "</dl>",
        _optional_reading("קריאה שלישית של GPT", candidate.gpt_third_reading),
        _optional_reading("קריאה שלישית של Grok", candidate.grok_third_reading),
        _optional_reading("בסיס היישוב", candidate.reconciliation_basis),
        _optional_reading("החלטת אתגור שנרשמה", candidate.challenge_decision),
        "<h3>ראיות</h3>",
        _evidence_html(candidate, crop_paths),
        "<h3>קריאות שהושמו בצד</h3>",
        _superseded_html(item.superseded),
        "</details>",
        "</div>",
        '<div class="viewport">',
        f'<div class="sheet" data-native-width="{page.width}" '
        f'data-native-height="{page.height}" style="width:100%">',
        f'<img class="page" src="{_esc(relative)}" '
        f'width="{page.width}" height="{page.height}" '
        f'alt="עמוד מקורי, מסמך {candidate.document_id}, עמוד {_esc(page_label)}">',
        _overlay(candidate, page),
        _human_overlay(mark),
        _proposal_overlay(proposal),
        _confirmed_overlay(confirmed),
        "</div>",
        "</div>",
        "</div>",
        "</article>",
    ]
    return "\n".join(parts)


_CATEGORY_HEBREW: Final[dict[str, str]] = {
    "MISSING_CONTENT": "אחת הקריאות כוללת טקסט שהקריאה השנייה לא כללה.",
    "READING_ORDER": "המחלוקת היא על סדר הקריאה.",
    "NAME_OR_ENTITY": "המחלוקת היא על שם או על גוף.",
    "DATE_OR_NUMBER": "המחלוקת היא על תאריך או על מספר.",
    "WORDING": "המחלוקת היא על ניסוח.",
    "CANCELLATION_OR_INSERTION": "המחלוקת היא על מחיקה או על הוספה.",
    "PUNCTUATION": "המחלוקת היא על פיסוק.",
    "COVERAGE_UNCERTAINTY": "המחלוקת היא על כיסוי הטקסט בעמוד.",
}


def _category_label(category: str) -> str:
    return _CATEGORY_HEBREW.get(category, category)


def _page_label(page_id: str) -> str:
    if page_id.startswith("page_") and page_id.removeprefix("page_").isdigit():
        return str(int(page_id.removeprefix("page_")))
    return page_id


def _quote(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _disagreement_html(candidate: LearningCandidate) -> str:
    gpt = candidate.gpt_text.strip()
    grok = candidate.grok_text.strip()
    hint = _category_label(candidate.dispute_category)
    if gpt and not grok:
        sentence = (
            f"GPT כלל את הקטע «{_quote(gpt)}». "
            "Grok לא כלל קטע במקום הזה. "
            "בדקו בעמוד אם הקטע מופיע ומה כתוב בו."
        )
    elif grok and not gpt:
        sentence = (
            f"Grok כלל את הקטע «{_quote(grok)}». "
            "GPT לא כלל קטע במקום הזה. "
            "בדקו בעמוד אם הקטע מופיע ומה כתוב בו."
        )
    elif not gpt and not grok:
        sentence = "שתי הקריאות ריקות בקטע הזה. בדקו את העמוד."
    elif gpt == grok:
        sentence = (
            f"{hint} שתי הקריאות מציגות אותו טקסט: «{_quote(gpt)}». "
            "בדקו בעמוד אם זה מה שכתוב."
        )
    else:
        sentence = (
            f"{hint} GPT קרא: «{_quote(gpt)}». "
            f"Grok קרא: «{_quote(grok)}». "
            "בדקו בעמוד מה כתוב."
        )
    sentence += " אין כאן קביעה איזו קריאה נכונה."
    return f'<section class="disagreement"><p>{_esc(sentence)}</p></section>'


def _where_sentence(
    proposal: VisualProposal | None, confirmed: ConfirmedLocationDisplay | None
) -> str:
    ambiguous = proposal is not None and proposal.status == "AMBIGUOUS"
    missing = proposal is not None and proposal.status in {"NOT_FOUND", "ERROR"}
    suggested = proposal is not None and bool(proposal.boxes) and not missing
    located = confirmed is not None and bool(confirmed.boxes)
    if ambiguous:
        return "סומנו כמה הצעות מיקום, ואף אחת לא נבחרה. ההצעות לא מאשרות את הקריאה."
    if suggested and located:
        return (
            "המסגרת הכחולה היא הצעת מיקום שלא אומתה. "
            "המסגרת הירוקה היא מיקום שאדם אישר בעמוד. "
            "אישור המיקום אינו אישור של הקריאה."
        )
    if suggested:
        return "המסגרת הכחולה היא הצעת מיקום. היא לא אומתה ואינה קובעת מה כתוב."
    if located:
        return "המסגרת הירוקה היא מיקום שאדם אישר בעמוד. הקריאה עצמה לא אומתה."
    if missing and proposal is not None and proposal.status == "NOT_FOUND":
        return "לא נמצא אזור מתאים. עיינו בעמוד כולו."
    return "אין מיקום מסומן. עיינו בעמוד כולו."


def _location_sentence(mark: LocalizationMark | None) -> str:
    if mark is None:
        return "לא סומן מיקום."
    if mark.status == "HUMAN_MARKED":
        return "סימון אנושי בטאב זה. מקור: HUMAN_MARKED. התמלול נשאר NOT_VERIFIED."
    if mark.status == "UNCLEAR":
        return "המיקום לא ברור. לא נשמרה תיבה. התמלול נשאר NOT_VERIFIED."
    return "אין מיקום. לא נשמרה תיבה. התמלול נשאר NOT_VERIFIED."


def _proposal_sentence(proposal: VisualProposal | None) -> str:
    if proposal is None:
        return "לא נרשמה הצעת מיקום."
    if proposal.status == "PROPOSED":
        sentence = "המודל הציע מיקום. MODEL_PROPOSED. זה לא אימות של התמלול."
    elif proposal.status == "AMBIGUOUS":
        sentence = "המודל הציע יותר מאזור אחד ולא בחר ביניהם. זה לא אימות של התמלול."
    elif proposal.status == "NOT_FOUND":
        sentence = "המודל לא מצא אזור. זו הימנעות, לא כשל. התמלול נשאר NOT_VERIFIED."
    else:
        sentence = "בקשת המיקום נכשלה. אין תיבה. התמלול נשאר NOT_VERIFIED."
    if proposal.explanation:
        sentence += " הסבר שלא אומת: " + proposal.explanation
    if proposal.nearby_text_anchor:
        sentence += " טקסט סמוך שלא אומת: " + proposal.nearby_text_anchor
    return sentence


def _proposal_overlay(proposal: VisualProposal | None) -> str:
    if proposal is None or not proposal.boxes:
        return ""
    if proposal.location_authority != "MODEL_PROPOSED":
        raise ReviewIntegrityError("a drawn suggestion must stay a model proposal")
    parts: list[str] = []
    several = len(proposal.boxes) > 1
    for index, box in enumerate(proposal.boxes, start=1):
        label = f"הצעה {index}" if several else "הצעת מיקום"
        parts.append(
            '<div class="proposal" data-provenance="MODEL_PROPOSED" '
            f'data-region="{index - 1}" '
            f'data-nx="{box.x}" data-ny="{box.y}" '
            f'data-nw="{box.width}" data-nh="{box.height}" '
            f'style="left:{_norm_percent(box.x)}%;top:{_norm_percent(box.y)}%;'
            f'width:{_norm_percent(box.width)}%;height:{_norm_percent(box.height)}%">'
            f'<span class="proposal-caption">{_esc(label)}</span></div>'
        )
    return "\n".join(parts)


def _confirmed_overlay(confirmed: ConfirmedLocationDisplay | None) -> str:
    if confirmed is None:
        return ""
    parts: list[str] = []
    several = len(confirmed.boxes) > 1
    for index, box in enumerate(confirmed.boxes, start=1):
        label = f"מיקום שאושר {index}" if several else "מיקום שאושר"
        parts.append(
            '<div class="confirmed" data-provenance="HUMAN_CONFIRMED_LOCATION" '
            f'data-region="{index - 1}" '
            f'data-nx="{box.x}" data-ny="{box.y}" '
            f'data-nw="{box.width}" data-nh="{box.height}" '
            f'style="left:{_norm_percent(box.x)}%;top:{_norm_percent(box.y)}%;'
            f'width:{_norm_percent(box.width)}%;height:{_norm_percent(box.height)}%">'
            f'<span class="confirmed-caption">{_esc(label)}</span></div>'
        )
    return "\n".join(parts)


def _inspect_buttons(
    proposal: VisualProposal | None, confirmed: ConfirmedLocationDisplay | None
) -> list[str]:
    buttons: list[str] = []
    if proposal is not None:
        for index in range(len(proposal.boxes)):
            label = f"הצעה {index + 1}"
            buttons.append(
                '<button type="button" class="inspect-region" '
                f'data-kind="proposal" data-region="{index}">{label}</button>'
            )
    if confirmed is not None:
        for index in range(len(confirmed.boxes)):
            label = f"מיקום שאושר {index + 1}"
            buttons.append(
                '<button type="button" class="inspect-region" '
                f'data-kind="confirmed" data-region="{index}">{label}</button>'
            )
    return buttons


def _human_overlay(mark: LocalizationMark | None) -> str:
    if mark is None or mark.status != "HUMAN_MARKED" or mark.bbox is None:
        return ""
    box = mark.bbox
    left = _norm_percent(box.x)
    top = _norm_percent(box.y)
    width = _norm_percent(box.width)
    height = _norm_percent(box.height)
    return (
        '<div class="mark human" data-provenance="HUMAN_MARKED" '
        f'style="left:{left}%;top:{top}%;width:{width}%;height:{height}%"></div>'
    )


def _norm_percent(part: int) -> str:
    whole, frac = divmod(part, 10_000)
    return f"{whole}.{frac:04d}"


def _json_for_html(sidecar: LocalizationSidecar) -> str:
    payload = sidecar.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")


def bind_sidecar_to_cases(
    sidecar: LocalizationSidecar, cases: Sequence[QueuedCase]
) -> LocalizationSidecar:
    return bind_sidecar(sidecar, [_binding(item) for item in cases])


def bind_proposals_to_cases(
    sidecar: VisualProposalSidecar,
    cases: Sequence[QueuedCase],
    *,
    export_sha256: str | None,
) -> VisualProposalSidecar:
    return bind_visual_proposals(
        sidecar,
        [_proposal_binding(item) for item in cases],
        export_sha256=export_sha256,
    )


def _binding(item: QueuedCase) -> CaseBinding:
    candidate = item.candidate
    return CaseBinding(
        candidate_id=candidate.candidate_id,
        document_id=candidate.document_id,
        page_id=candidate.page_id,
        page_image_sha256=candidate.page_image_sha256,
        page_width=item.page.width,
        page_height=item.page.height,
        dispute_category=candidate.dispute_category,
        pilot=is_pilot_dispute(
            candidate.document_id, candidate.page_id, candidate.dispute_category
        ),
    )


def _proposal_binding(item: QueuedCase) -> ProposalBinding:
    candidate = item.candidate
    return ProposalBinding(
        candidate_id=candidate.candidate_id,
        dispute_id=candidate.dispute_id,
        document_id=candidate.document_id,
        page_id=candidate.page_id,
        page_image_sha256=candidate.page_image_sha256,
        page_width=item.page.width,
        page_height=item.page.height,
        dispute_category=candidate.dispute_category,
        pilot=is_pilot_dispute(
            candidate.document_id, candidate.page_id, candidate.dispute_category
        ),
    )


def _require_unique_pilots(cases: Sequence[QueuedCase]) -> None:
    for document_id, page_id, category in PILOT_DISPUTES:
        hits = [
            item
            for item in cases
            if item.candidate.document_id == document_id
            and item.candidate.page_id == page_id
            and item.candidate.dispute_category == category
        ]
        if len(hits) > 1:
            raise ReviewIntegrityError(
                f"pilot dispute {document_id} {page_id} {category} is not unique"
            )


def _overlay(candidate: LearningCandidate, page: PageRef) -> str:
    box = candidate.image_bbox
    if box is None:
        return ""
    frame = BBox(x=0, y=0, width=page.width, height=page.height)
    if not frame.contains(box):
        raise ReviewIntegrityError(
            f"recorded box for {candidate.candidate_id} does not lie on the page"
        )
    left = _percent(box.x, page.width)
    top = _percent(box.y, page.height)
    width = _percent(box.width, page.width)
    height = _percent(box.height, page.height)
    return f'<div class="box" style="left:{left}%;top:{top}%;width:{width}%;height:{height}%"></div>'


def _evidence_html(candidate: LearningCandidate, crop_paths: Mapping[str, str]) -> str:
    if not candidate.evidence:
        return "<p>לא נרשמה הפניה לראיה.</p>"
    lines = ["<ul>"]
    for item in candidate.evidence:
        window = item.window_kind or "not recorded"
        detail = (
            f"<li>{_esc(item.provider)} {_esc(item.kind)} "
            f"<code>{_esc(item.evidence_id)}</code> "
            f"assessment {_esc(item.assessment or 'not recorded')} "
            f"window {_esc(window)} exact text coordinates false"
        )
        if item.window_kind == "FULL_PAGE_FALLBACK":
            detail += (
                ". חלופה של עמוד מלא. Full-page fallback. This is not a word crop."
            )
        elif item.window_kind == "TARGETED" and item.crop_sha256:
            relative = crop_paths.get(item.crop_sha256)
            if relative is None:
                raise ReviewIntegrityError(
                    f"targeted crop {item.crop_sha256} is not in the packet"
                )
            detail += (
                f'. <img class="crop" src="{_esc(relative)}" '
                f'alt="Recorded targeted crop {_esc(item.crop_sha256)}">'
            )
        elif item.window_kind is None:
            detail += ". No crop image is associated with this evidence reference."
        lines.append(detail + "</li>")
    lines.append("</ul>")
    return "\n".join(lines)


def _superseded_html(rows: Sequence[LearningCandidate]) -> str:
    if not rows:
        return "<p>אין קריאה שהושמה בצד במקרה הזה.</p>"
    lines = ["<ul>"]
    for item in rows:
        lines.append(
            f"<li><code>{_esc(item.candidate_id)}</code> NOT_VERIFIED. Set aside: "
            f"{_reading(item.superseded_text or '')}</li>"
        )
    lines.append("</ul>")
    return "\n".join(lines)


def _reading(text: str) -> str:
    if text == "":
        return '<p class="empty">לא נמסרה קריאה.</p>'
    return f'<pre class="reading">{_esc(text)}</pre>'


def _optional_reading(title: str, text: str | None) -> str:
    if text is None:
        return ""
    return f"<h3>{_esc(title)}</h3>\n{_reading(text)}"


def _row(label: str, value: str) -> str:
    return f"<dt>{_esc(label)}</dt><dd>{_esc(value)}</dd>"


def _percent(part: int, whole: int) -> str:
    scaled = (part * 100_000_000) // whole
    return f"{scaled // 1_000_000}.{scaled % 1_000_000:06d}"


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _sort_key(item: QueuedCase) -> tuple[object, ...]:
    candidate = item.candidate
    try:
        category_rank = DISPUTE_PRIORITY[DisputeCategory(candidate.dispute_category)]
    except ValueError as exc:
        raise ReviewIntegrityError(
            f"candidate {candidate.candidate_id} has an unknown dispute category"
        ) from exc
    return (
        item.band,
        _span_rank(candidate.image_span_status),
        category_rank,
        max(len(candidate.gpt_text), len(candidate.grok_text)),
        candidate.document_id,
        item.page.reading_order,
        candidate.page_id,
        candidate.dispute_id,
        candidate.candidate_id,
    )


def _span_rank(status: str) -> int:
    if status in _LOCATED_SPANS:
        return 0
    if status in _UNLOCATED_SPANS:
        return 1
    raise ReviewIntegrityError(f"unknown image span status {status}")


def _require_box_matches_status(candidate: LearningCandidate) -> None:
    located = candidate.image_span_status in _LOCATED_SPANS
    if located != (candidate.image_bbox is not None):
        raise ReviewIntegrityError(
            f"image box for {candidate.candidate_id} does not match its span status"
        )


def _page_index(
    export: LearningCandidateExport,
) -> dict[tuple[str, int, str], PageRef]:
    found: dict[tuple[str, int, str], PageRef] = {}
    for document in export.documents:
        for page in document.pages:
            key = (document.batch_id, document.document_id, page.page_id)
            if key in found and found[key] != page:
                raise ReviewIntegrityError("export page identity is not unique")
            found[key] = page
    return found


def _superseded_by_primary(
    export: LearningCandidateExport,
) -> dict[str, tuple[LearningCandidate, ...]]:
    grouped: dict[str, list[LearningCandidate]] = {}
    for candidate in export.candidates:
        if candidate.reading_role != "SUPERSEDED_READING":
            continue
        if candidate.related_candidate_id is None:
            continue
        grouped.setdefault(candidate.related_candidate_id, []).append(candidate)
    return {
        key: tuple(sorted(rows, key=lambda item: item.candidate_id))
        for key, rows in grouped.items()
    }


def _find_targeted_crop(
    review_dirs: Sequence[Path],
    candidate: LearningCandidate,
    evidence: EvidenceRefRecord,
) -> LocatedCrop:
    matches: list[LocatedCrop] = []
    for reviews in review_dirs:
        root = reviews_root(reviews)
        document = (
            root
            / f"batch_{candidate.batch_id}"
            / "documents"
            / f"document_{candidate.document_id}"
        )
        if not document.is_dir():
            continue
        for record_path in sorted(
            document.glob("challenge_round_*/input/crops/*.json")
        ):
            resolved = contained(document, record_path)
            parts = resolved.relative_to(document).parts
            if (
                len(parts) != 4
                or not parts[0].startswith("challenge_round_")
                or parts[1:3] != ("input", "crops")
            ):
                continue
            try:
                crop = CropRecord.model_validate_json(resolved.read_bytes())
            except ValueError as exc:
                raise ReviewIntegrityError(
                    "challenge crop does not match the schema"
                ) from exc
            if crop.crop_sha256 != evidence.crop_sha256:
                continue
            if crop.window_kind != "TARGETED":
                raise ReviewIntegrityError(
                    "targeted evidence names a crop that is not a targeted window"
                )
            if (
                crop.page_id != candidate.page_id
                or crop.page_sha256 != candidate.page_image_sha256
            ):
                raise ReviewIntegrityError(
                    "targeted crop does not belong to the candidate page"
                )
            if crop.crop_png_width is None or crop.crop_png_height is None:
                raise ReviewIntegrityError("targeted crop has no recorded pixel size")
            image_name = _one_segment(crop.image_filename or f"{crop.crop_id}.png")
            image = contained(resolved.parent, resolved.parent / image_name)
            if not image.is_file():
                raise ReviewIntegrityError(f"missing targeted crop image {image_name}")
            if sha256_bytes(image.read_bytes()) != evidence.crop_sha256:
                raise ReviewIntegrityError(
                    "targeted crop hash does not match the evidence"
                )
            matches.append(
                LocatedCrop(
                    evidence=evidence,
                    path=image,
                    width=crop.crop_png_width,
                    height=crop.crop_png_height,
                )
            )
    if not matches:
        raise ReviewIntegrityError(
            f"missing targeted crop for candidate {candidate.candidate_id}"
        )
    first = matches[0].path.read_bytes()
    for other in matches[1:]:
        if other.path.read_bytes() != first:
            raise ReviewIntegrityError("targeted crop copies disagree")
    return matches[0]


def _image_relative(digest: str, mime_type: str) -> str:
    try:
        suffix = _IMAGE_SUFFIX[mime_type]
    except KeyError as exc:
        raise ReviewIntegrityError(f"unsupported page image type {mime_type}") from exc
    return f"images/{digest}{suffix}"


def _one_segment(value: str) -> str:
    if not value or "/" in value or "\\" in value or value in {".", ".."}:
        raise ReviewPathError("image filename must be one path segment")
    return value


def _unique_ids(values: Iterable[int]) -> list[int]:
    found: list[int] = []
    for value in values:
        if value not in found:
            found.append(value)
    return found


def _unique_text(values: Iterable[str]) -> list[str]:
    found: list[str] = []
    for value in values:
        if value not in found:
            found.append(value)
    return found


def _refuse_sidecar_path(
    path: Path, review_dirs: Sequence[Path], batches_dir: Path | None
) -> None:
    resolved = path.resolve()
    protected = [reviews_root(item) for item in review_dirs]
    if batches_dir is not None:
        protected.append(reviews_root(batches_dir))
    for root in protected:
        root_resolved = root.resolve()
        if resolved == root_resolved or root_resolved in resolved.parents:
            raise ReviewPathError(
                "refusing to read a visual proposal sidecar from a review or sealed bundle"
            )


def _refuse_protected(
    output_dir: Path, review_dirs: Sequence[Path], batches_dir: Path | None
) -> None:
    resolved = output_dir.resolve()
    protected = list(review_dirs)
    if batches_dir is not None:
        protected.append(batches_dir)
    for root in protected:
        root_resolved = root.resolve()
        if resolved == root_resolved or root_resolved in resolved.parents:
            raise ReviewPathError(
                "refusing to write the packet inside a review or sealed bundle"
            )
        if resolved in root_resolved.parents:
            raise ReviewPathError(
                "refusing to write the packet over a review or sealed bundle"
            )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReviewUsageError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    except ReviewError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
