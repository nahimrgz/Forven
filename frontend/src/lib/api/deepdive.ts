import { fetchApi, fetchApiStream } from './core';

export type DeepdiveThread = {
	id: string;
	strategy_id: string;
	created_at: string;
	updated_at: string;
	archived_at: string | null;
};

export type DeepdiveMessageRole = 'user' | 'assistant' | 'tool';

export type DeepdiveToolCall = {
	name: string;
	input?: Record<string, unknown>;
	id?: string;
};

export type DeepdiveMessage = {
	id: string;
	thread_id: string;
	role: DeepdiveMessageRole;
	content: string;
	tool_call: DeepdiveToolCall | null;
	created_at: string;
	cost_usd: number | null;
	model: string | null;
};

export type DeepdiveStreamEvent =
	| { type: 'user_persisted' }
	| { type: 'assistant_token'; content: string }
	| { type: 'tool_call'; name: string; input: Record<string, unknown> }
	| { type: 'tool_result'; name: string; output: string }
	| { type: 'done'; message_id: string }
	| { type: 'error'; code: string; message: string };

export async function createOrGetDeepdiveThread(strategyId: string): Promise<DeepdiveThread> {
	return fetchApi<DeepdiveThread>('/deepdive/threads', {
		method: 'POST',
		body: JSON.stringify({ strategy_id: strategyId }),
	});
}

export async function archiveDeepdiveThread(threadId: string): Promise<void> {
	await fetchApi<{ ok: boolean }>(`/deepdive/threads/${encodeURIComponent(threadId)}/archive`, {
		method: 'POST',
	});
}

export async function listDeepdiveMessages(threadId: string): Promise<DeepdiveMessage[]> {
	const resp = await fetchApi<{ messages: DeepdiveMessage[] }>(
		`/deepdive/threads/${encodeURIComponent(threadId)}/messages`,
	);
	return resp.messages;
}

export async function streamDeepdiveSend(
	threadId: string,
	userText: string,
	onEvent: (event: DeepdiveStreamEvent) => void,
): Promise<void> {
	const r = await fetchApiStream(`/deepdive/threads/${encodeURIComponent(threadId)}/send`, {
		method: 'POST',
		body: JSON.stringify({ user_text: userText }),
	});
	if (!r.body) {
		throw new Error('deepdive send failed: response stream missing');
	}
	const reader = r.body.getReader();
	const decoder = new TextDecoder();
	let buffer = '';
	while (true) {
		const { value, done } = await reader.read();
		if (done) break;
		buffer += decoder.decode(value, { stream: true });
		let sepIdx = buffer.indexOf('\n\n');
		while (sepIdx !== -1) {
			const block = buffer.slice(0, sepIdx);
			buffer = buffer.slice(sepIdx + 2);
			const dataLine = block.split('\n').find((line) => line.startsWith('data: '));
			if (dataLine) {
				try {
					const parsed = JSON.parse(dataLine.slice('data: '.length)) as DeepdiveStreamEvent;
					onEvent(parsed);
				} catch {
					// malformed event — ignore
				}
			}
			sepIdx = buffer.indexOf('\n\n');
		}
	}
}
