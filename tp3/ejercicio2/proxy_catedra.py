"""Ejecuta y registra una llamada a GPT-4o por el proxy de la cátedra."""

from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

from dotenv import dotenv_values
from openai import APIConnectionError, APIStatusError, OpenAI


MODELO_PROXY = "openai/gpt-4o-2024-08-06"
HOST_PROXY = "sa-llmproxy.it.itba.edu.ar"


def completar(mensajes: list[dict], max_tokens: int, formato_json: bool = False) -> dict:
    entorno = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
    base = entorno.get("OPENAI_BASE_URL") or f"https://{HOST_PROXY}/"
    url = urlsplit(base)
    if url.scheme != "https" or url.netloc != HOST_PROXY or url.query or url.fragment:
        raise ValueError("OPENAI_BASE_URL debe apuntar al proxy HTTPS de la cátedra.")
    if not entorno.get("OPENAI_API_KEY"):
        raise ValueError("Falta OPENAI_API_KEY en tp3/.env.")

    cliente = OpenAI(base_url=base, api_key=entorno["OPENAI_API_KEY"], max_retries=0, timeout=60)
    opciones = {"response_format": {"type": "json_object"}} if formato_json else {}
    inicio = perf_counter()
    try:
        http = cliente.chat.completions.with_raw_response.create(
            model=MODELO_PROXY, messages=mensajes, temperature=0, store=False,
            max_completion_tokens=max_tokens, **opciones,
        )
    except APIStatusError as error:
        raise RuntimeError(f"El proxy devolvió HTTP {error.status_code}.") from error
    except APIConnectionError as error:
        raise RuntimeError("La llamada al proxy no devolvió una respuesta confirmada.") from error

    respuesta = http.parse()
    opcion = respuesta.choices[0]
    return {
        "modelo_solicitado": MODELO_PROXY, "modelo": respuesta.model, "mensajes": mensajes,
        "temperature": 0, "store": False, "max_completion_tokens": max_tokens,
        "response_format": "json_object" if formato_json else "text",
        "finish_reason": opcion.finish_reason, "segundos": perf_counter() - inicio,
        "uso": respuesta.usage.model_dump() if respuesta.usage else None,
        "costo_reportado_proxy": http.headers.get("x-litellm-response-cost"),
        "respuesta": opcion.message.content,
    }
