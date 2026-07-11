import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import * as api from '../lib/api';
import {
	createOrGetAssistantThread,
	confirmAssistantAction,
	listAssistantMessages,
	streamAssistantSend,
	type AssistantStreamEvent,
} from '../lib/api/assistant';

const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('assistant api client', () => {
	beforeEach(() => {
		mockFetch.mockReset();
		window.localStorage.clear();
	});
	afterEach(() => {
		vi.clearAllMocks();
	});

	it('re-exports through the api barrel', () => {
		expect(typeof api.createOrGetAssistantThread).toBe('function');
		expect(typeof api.streamAssistantSend).toBe('function');
		expect(typeof api.confirmAssistantAction).toBe('function');
	});

	it('createOrGetAssistantThread posts the scope', async () => {
		mockFetch.mockResolvedValueOnce({
			ok: true,
			json: () => Promise.resolve({ id: 'as_1', scope_kind: 'global', scope_id: null }),
		});
		const t = await createOrGetAssistantThread({ pageRoute: '/lab' });
		expect(t.id).toBe('as_1');
		const [url, init] = mockFetch.mock.calls[0];
		expect(String(url)).toContain('/api/assistant/threads');
		expect(JSON.parse(String((init as RequestInit).body))).toEqual({
			scope_kind: 'global',
			scope_id: null,
			page_route: '/lab',
		});
	});

	it('listAssistantMessages unwraps the messages array', async () => {
		mockFetch.mockResolvedValueOnce({
			ok: true,
			json: () => Promise.resolve({ messages: [{ id: 'm1', role: 'user', content: 'hi' }] }),
		});
		const got = await listAssistantMessages('as_1');
		expect(got).toHaveLength(1);
		expect(got[0].id).toBe('m1');
	});

	it('confirmAssistantAction posts approve flag', async () => {
		mockFetch.mockResolvedValueOnce({
			ok: true,
			json: () => Promise.resolve({ ok: true, status: 'executed', message: 'done' }),
		});
		const r = await confirmAssistantAction('as_1', 'asm_9', true);
		expect(r.status).toBe('executed');
		const [url, init] = mockFetch.mock.calls[0];
		expect(String(url)).toContain('/api/assistant/threads/as_1/actions/asm_9/confirm');
		expect(JSON.parse(String((init as RequestInit).body))).toEqual({ approve: true });
	});

	it('streamAssistantSend parses SSE incl. action_proposed and sends context', async () => {
		window.localStorage.setItem('forven_api_key', 'assistant-api-key');
		window.localStorage.setItem('forven_operator_key', 'assistant-operator-key');
		const encoder = new TextEncoder();
		const chunks = [
			encoder.encode('data: {"type":"user_persisted"}\n\n'),
			encoder.encode(
				'data: {"type":"assistant_token","content":"proposing"}\n\n' +
					'data: {"type":"action_proposed","action_id":"a1","name":"promote_strategy","input":{},"summary":"Promote S1"}\n\n' +
					'data: {"type":"done","message_id":"m1"}\n\n',
			),
		];
		let i = 0;
		const reader = {
			read: vi.fn().mockImplementation(async () =>
				i < chunks.length ? { value: chunks[i++], done: false } : { value: undefined, done: true },
			),
		};
		mockFetch.mockResolvedValueOnce({ ok: true, body: { getReader: () => reader } });

		const events: AssistantStreamEvent[] = [];
		await streamAssistantSend('as_1', 'promote S1', { route: '/lab', page_kind: 'lab' }, (e) => events.push(e), true);

		expect(events.map((e) => e.type)).toEqual(['user_persisted', 'assistant_token', 'action_proposed', 'done']);
		const [url, init] = mockFetch.mock.calls[0];
		expect(String(url)).toContain('/api/assistant/threads/as_1/send');
		const body = JSON.parse(String((init as RequestInit).body));
		const headers = (init as RequestInit).headers as Headers;
		expect(headers.get('X-API-Key')).toBe('assistant-api-key');
		expect(headers.get('X-Operator-Key')).toBe('assistant-operator-key');
		expect(body.user_text).toBe('promote S1');
		expect(body.allow_actions).toBe(true);
		expect(body.page_context).toEqual({ route: '/lab', page_kind: 'lab' });
	});

	it('streamAssistantSend throws on non-2xx', async () => {
		mockFetch.mockResolvedValueOnce({ ok: false, status: 409, body: null });
		await expect(
			streamAssistantSend('as_1', 'x', null, () => {}, true),
		).rejects.toThrow();
	});
});
