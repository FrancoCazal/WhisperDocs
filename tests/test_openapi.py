"""Every operation's ID, fixed (audit R20).

Clients name the API's calls by these IDs: the MCP server, a generated SDK, a
tool list for an agent. FastAPI's default, from the function name and the
path (`query_api_v1_query_post`), changes with any rename or move. Each route
sets its own; changing one breaks those clients, so it changes here too, on
purpose.
"""

from app.main import app

OPERATIONS = {
    "GET /api/v1/health": "check_health",
    "POST /api/v1/webhook/whatsapp": "receive_whatsapp_message",
    "POST /api/v1/webhook/whatsapp/status": "receive_whatsapp_status",
    "GET /api/v1/metrics": "get_metrics",
    "GET /api/v1/admin/settings": "get_settings",
    "GET /api/v1/admin/model-status": "get_model_status",
    "GET /api/v1/admin/audit-log": "list_audit_log",
    "GET /api/v1/collections": "list_collections",
    "POST /api/v1/collections": "create_collection",
    "GET /api/v1/collections/{collection_id}": "get_collection",
    "DELETE /api/v1/collections/{collection_id}": "delete_collection",
    "POST /api/v1/documents/upload": "upload_document",
    "POST /api/v1/documents/upload-batch": "upload_documents_batch",
    "GET /api/v1/documents": "list_documents",
    "GET /api/v1/documents/{document_id}": "get_document",
    "DELETE /api/v1/documents/{document_id}": "delete_document",
    "GET /api/v1/documents/{document_id}/chunks": "list_document_chunks",
    "POST /api/v1/documents/bulk-delete": "bulk_delete_documents",
    "POST /api/v1/documents/bulk-move": "bulk_move_documents",
    "POST /api/v1/documents/upload-async": "upload_document_async",
    "GET /api/v1/documents/jobs/{job_id}": "get_ingestion_job",
    "GET /api/v1/documents/jobs": "list_ingestion_jobs",
    "GET /api/v1/documents/jobs/{job_id}/stream": "stream_ingestion_job",
    "POST /api/v1/query": "answer_question",
    "GET /api/v1/query/history": "list_query_history",
    "POST /api/v1/retrieve": "retrieve_chunks",
    "POST /api/v1/eval/run": "run_eval",
    "GET /api/v1/eval/results": "list_eval_results",
    "GET /api/v1/eval/results/{run_id}": "get_eval_result",
    "GET /api/v1/eval/diff": "diff_eval_results",
    "POST /api/v1/agent/chat": "agent_chat",
    "POST /api/v1/agent/chat/stream": "agent_chat_stream",
    "GET /api/v1/conversations": "list_conversations",
    "GET /api/v1/conversations/{thread_id}": "get_conversation",
    "PATCH /api/v1/conversations/{thread_id}": "update_conversation",
    "DELETE /api/v1/conversations/{thread_id}": "delete_conversation",
    "GET /api/v1/conversations/{thread_id}/messages": "list_conversation_messages",
    "POST /api/v1/conversations/{thread_id}/notes": "create_note",
    "GET /api/v1/notes": "list_notes",
    "DELETE /api/v1/notes/{note_id}": "delete_note",
}


def test_every_operation_has_its_own_fixed_id():
    operations = {
        f"{method.upper()} {path}": op["operationId"]
        for path, ops in app.openapi()["paths"].items()
        for method, op in ops.items()
    }

    assert operations == OPERATIONS
    assert len(set(operations.values())) == len(operations)
