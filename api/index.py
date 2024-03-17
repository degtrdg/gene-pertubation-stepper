from typing import List, Dict
import json
from uuid import uuid4
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.help.main import ProteinGraph, ProteinExplorerAgent


class InitializeRequest(BaseModel):
    start_protein: str
    perturbation: str
    target_condition: str


class ExploreNextRequest(BaseModel):
    session_id: str


class VisualizeRequest(BaseModel):
    session_id: str


app = FastAPI(docs_url="/api/docs", openapi_url="/api/openapi.json")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session state to store the state of the ProteinGraph and ProteinExplorerAgent
session_state = {}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/api/initialize")
def initialize_protein_explorer(request: InitializeRequest):
    try:
        session_id = str(uuid4())  # Generate a unique session ID
        info_file_path = './api/help/9606.protein.info.v12.0.txt'
        links_file_path = './api/help/top_protein_links.csv'
        protein_graph = ProteinGraph(
            info_file_path, links_file_path, request.start_protein)
        protein_explorer = ProteinExplorerAgent(
            protein_graph, request.start_protein, request.perturbation, request.target_condition)
        session_state[session_id] = {
            "protein_graph": protein_graph,
            "protein_explorer": protein_explorer
        }
        session = session_state[session_id]
        start, _ = session["protein_explorer"].explore(
            start_protein=request.start_protein)
        # Return the initial state and visualization data
        nodes, edges = session["protein_graph"].get_graph_data_for_visualization(
        )
        return {"session_id": session_id, "nodes": nodes, "edges": edges}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/start/{session_id}")
async def start_protein_explorer(session_id: str):
    try:
        session = session_state.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")

        start_protein = session["protein_graph"].start_protein
        _, edges = session["protein_graph"].get_graph_data_for_visualization()

        def convert_edges_format(edges):
            protein1_list = [edge['source'] for edge in edges]
            protein2_list = [edge['target'] for edge in edges]
            return {'protein1': protein1_list, 'protein2': protein2_list}

        data = await session["protein_explorer"].start(
            start_protein=start_protein, edges=convert_edges_format(edges))

        messages = session['protein_explorer'].context.messages

        async def generate(messages):
            chunks = []
            async for chunk in data:
                if chunk is not None:
                    # Encode the newline characters within the content
                    content = chunk.choices[0].delta.content.replace(
                        '\n', '\\m') if chunk.choices[0].delta.content else ''
                    chunks.append(chunk.choices[0].delta.content)
                    yield f"data: {content}\n\n"
            # final chunk should have whole message
            # Ensure all elements are strings and join them
            chunks = [str(chunk) for chunk in chunks if chunk is not None]
            messages.append(
                {'role': 'assistant', 'content': ''.join(chunks)})
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(messages), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/simulate_change/{session_id}/{next_protein}")
async def simulate_change(session_id: str, next_protein: str):
    try:
        session_id = session_id
        session = session_state.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")

        data = await session["protein_explorer"].simulate_state_change(
            protein=next_protein)

        messages = session['protein_explorer'].context.messages

        async def generate(messages):
            chunks = []
            async for chunk in data:
                if chunk is not None:
                    # Encode the newline characters within the content
                    content = chunk.choices[0].delta.content.replace(
                        '\n', '\\m') if chunk.choices[0].delta.content else ''
                    chunks.append(chunk.choices[0].delta.content)
                    yield f"data: {content}\n\n"
            # final chunk should have whole message
            # Ensure all elements are strings and join them
            chunks = [str(chunk) for chunk in chunks if chunk is not None]
            messages.append(
                {'role': 'assistant', 'content': ''.join(chunks)})
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(messages), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/check_condition")
def check_condition(request: ExploreNextRequest):
    try:
        session_id = request.session_id
        session = session_state.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
        condition_met = session["protein_explorer"].check_for_condition()
        return {"condition_met": condition_met}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/visualize")
def visualize_protein_graph(request: VisualizeRequest):
    try:
        session_id = request.session_id
        session = session_state.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
        next_protein, edges = session["protein_explorer"].explore()
        nodes, edges = session["protein_graph"].get_graph_data_for_visualization(
        )
        return {"nodes": nodes, "edges": edges, "next_protein": next_protein}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
