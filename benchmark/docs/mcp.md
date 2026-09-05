# MCP Integration

The assistant exposes internal document tools to the local model through MCP,
the Model Context Protocol. This document explains the protocol, our server and
how to debug it.

## What MCP is

MCP is an open protocol that standardises how an application offers tools,
resources and prompts to a language model. Instead of hard coding one
integration per model runtime, the application runs an MCP server and any
compatible client can discover and call the exposed capabilities. Messages are
JSON-RPC 2.0 objects exchanged over stdio or HTTP.

## Our MCP server

The MCP server in `src/mcp/server.py` is built with FastMCP and speaks stdio. It
starts as a child process of the assistant and exposes three tools:

- `read_document(file_path)` returns the full text of one document
- `list_documents()` lists every indexed document
- `search_documents(query)` finds documents by name

The server refuses to read anything outside the configured documents directory,
which prevents path traversal through a crafted tool argument.

## Handshake

A session begins with an `initialize` request carrying the protocol version and
the client capabilities. The server answers with its own capabilities and the
list of tools it provides. Only after that exchange may the client send
`tools/call` requests. A client that skips the handshake receives an error and
the connection is closed.

## When the model should call a tool

Retrieval already puts the most relevant passages into the prompt, so a tool
call is only worth its latency when the retrieved passages are insufficient:
the user asks for a complete document, asks which documents exist, or refers to
a file by name that was not retrieved. Prefer answering from the retrieved
context when it is sufficient.

## Debugging

Run the server directly and send a handshake by hand to see the raw frames. If
the assistant reports that the tool process disconnected, the server usually
crashed during import; its traceback goes to stderr, which the client drains
separately from the protocol stream.

## Transports

Our server speaks stdio, which is the right choice for a tool that runs on the
same machine as the client: no port to allocate, no authentication to
configure, and the process lifetime is tied to the client. The HTTP transport
exists for remote servers; it needs a credential, and every request must be
authorised on its own because the connection carries no session identity.

## Tool design

A tool is described by a name, a human readable description and a JSON schema
for its arguments. The model chooses a tool from that description alone, so the
description is part of the interface: state what the tool returns and when it
should be used, not how it is implemented. Keep the argument list small and the
arguments primitive; a tool that takes a nested object is harder for a small
model to call correctly.

Return text that is useful to a model rather than to a human dashboard. A
listing of file names is better than a table with box drawing characters, and
an error should say what went wrong in one sentence so that the model can
recover instead of repeating the same call.

## Resources and prompts

Besides tools, a server may expose resources, which are read only pieces of
context addressed by URI, and prompts, which are reusable templates the user
can invoke. We only use tools today; resources would be a natural fit for
exposing the indexed documents once the corpus is large enough that listing
them in a tool response becomes unwieldy.

## Errors and timeouts

A tool that fails returns an error result rather than raising through the
transport, so the model can see the failure and react. The client applies a
timeout to every call; an unbounded tool call blocks the whole answer. When the
server needs longer than a few seconds it should return a partial result with
an explanation rather than holding the connection.

## Security considerations

Tool arguments come from a model and must be treated as untrusted input. Paths
are resolved and checked against the allowed root before any file is opened,
and the check happens on the resolved path so that `..` segments and symbolic
links cannot escape. Nothing in a tool result is executed. A tool that shells
out is a liability; prefer library calls with explicit arguments.

## Versioning

The protocol version is negotiated in the handshake. Adding a tool is backwards
compatible, removing one or changing the meaning of an argument is not - clients
cache the tool list for the lifetime of a session. When a tool has to change
incompatibly, add the new one alongside the old and remove the old one in a
later release.

## Testing

The server is tested by driving it through the same JSON-RPC frames the client
sends: a handshake, a `tools/list` and a set of `tools/call` requests covering
the happy path, a missing file and a path traversal attempt. Because the
transport is stdio, the whole test runs in process without a network.
