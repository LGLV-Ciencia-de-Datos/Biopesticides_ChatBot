import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from recommender import Recommender
from twilio.twiml.messaging_response import MessagingResponse

load_dotenv()
PORT = int(os.getenv("PORT", "8000"))

app = FastAPI(title="BioBot: asistente inteligente para biopesticidas (Versión 1.1)")
rec = Recommender()

class QueryIn(BaseModel):
    query: str
    k: int = 3

@app.get("/")
def root():
    return {"ok": True, "message": "BioBot API online"}

@app.post("/recommend")
def recommend(payload: QueryIn):
    hits = rec.search(payload.query, k=payload.k)
    out = []
    for sim, row in hits:
        row_out = {k: (row.get(k) if row.get(k) else "") for k in [
            "name","Description","Example pests controlled","Example applications","Uses",
            "Efficacy & activity","Canonical SMILES","Isomeric SMILES","Please cite as"
        ]}
        row_out["score"] = float(sim)
        out.append(row_out)
    return {"query": payload.query, "results": out}

# Webhook WhatsApp (Twilio)
@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    body = form.get("Body", "")
    k = 3
    hits = rec.search(body, k=k)

    resp = MessagingResponse()
    reply = "🔎 Recomendaciones de BioBot basadas en tu descripción:\n\n"
    for sim, row in hits:
        name = row.get("name", "")
        pests = row.get("Example pests controlled", "") or "—"
        apps  = row.get("Example applications", "") or "—"
        uses  = row.get("Uses", "") or "—"
        reply += f"• {name}\n   Plagas: {pests}\n   Aplicaciones: {apps}\n   Usos: {uses}\n\n"
    reply += "⚠️ Uso informativo; verifica regulaciones y etiqueta del producto en tu país."

    resp.message(reply[:1500])
    return PlainTextResponse(str(resp), media_type="application/xml")
