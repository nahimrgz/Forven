import { describe, expect, it } from 'vitest';
import { isLocalHost, preferredBrowserApiBase } from '../lib/api/core';

describe('isLocalHost', () => {
	it('treats localhost, loopback and empty host as local', () => {
		expect(isLocalHost('localhost')).toBe(true);
		expect(isLocalHost('127.0.0.1')).toBe(true);
		expect(isLocalHost('::1')).toBe(true);
		expect(isLocalHost('')).toBe(true);
		expect(isLocalHost('LOCALHOST')).toBe(true);
	});

	it('treats LAN IPs and dev-tunnel hosts as remote', () => {
		expect(isLocalHost('192.168.1.50')).toBe(false);
		expect(isLocalHost('3dm7hr0p-5173.usw3.devtunnels.ms')).toBe(false);
	});
});

describe('preferredBrowserApiBase', () => {
	it('uses the direct :8003 backend port on local hosts', () => {
		expect(
			preferredBrowserApiBase({
				protocol: 'http:',
				hostname: '127.0.0.1',
				origin: 'http://127.0.0.1:5173'
			})
		).toBe('http://127.0.0.1:8003/api');
		expect(
			preferredBrowserApiBase({
				protocol: 'http:',
				hostname: 'localhost',
				origin: 'http://localhost:5173'
			})
		).toBe('http://localhost:8003/api');
	});

	it('routes through the same-origin proxy on a dev tunnel (never :8003)', () => {
		const base = preferredBrowserApiBase({
			protocol: 'https:',
			hostname: '3dm7hr0p-5173.usw3.devtunnels.ms',
			origin: 'https://3dm7hr0p-5173.usw3.devtunnels.ms'
		});
		expect(base).toBe('https://3dm7hr0p-5173.usw3.devtunnels.ms/api');
		expect(base).not.toContain(':8003');
	});

	it('routes through the same-origin proxy on a LAN IP', () => {
		expect(
			preferredBrowserApiBase({
				protocol: 'http:',
				hostname: '192.168.1.50',
				origin: 'http://192.168.1.50:5173'
			})
		).toBe('http://192.168.1.50:5173/api');
	});
});
