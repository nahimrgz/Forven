import { describe, expect, it, vi } from 'vitest';

vi.mock('../lib/api/core', () => ({
	ACTIVE_API_BASE: 'http://127.0.0.1:8003/api',
	API_BASE: 'http://127.0.0.1:8003/api',
	isLocalHost: (hostname: string) =>
		['localhost', '127.0.0.1', '::1', ''].includes((hostname || '').toLowerCase()),
	fetchApi: vi.fn()
}));

import { getForvenLiveWebSocketUrls } from '../lib/api/forven';

describe('Forven websocket URL selection', () => {
	it('prefers the configured backend origin and avoids speculative fallbacks', () => {
		const urls = getForvenLiveWebSocketUrls();

		expect(urls[0]).toBe('ws://127.0.0.1:8003/api/ws/live');
		expect(urls).not.toContain('ws://127.0.0.1:8000/api/ws/live');
		expect(urls).not.toContain(`ws://${window.location.host}/api/ws/live`);
	});
});
