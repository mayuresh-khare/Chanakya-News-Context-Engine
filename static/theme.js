// CHANAKYA - Theme Toggle
// Persists the user's light/dark preference across page visits.

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    const btn = document.getElementById('theme-toggle-btn');
    if (btn) {
        btn.textContent = theme === 'dark' ? '☀️' : '🌙';
    }
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    const next = current === 'dark' ? 'light' : 'dark';
    localStorage.setItem('chanakya-theme', next);
    applyTheme(next);
}

// Apply saved theme immediately on page load (before paint, to avoid flash)
(function () {
    const saved = localStorage.getItem('chanakya-theme');
    if (saved) {
        applyTheme(saved);
    } else {
        // Respect system preference if no explicit choice saved yet
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        applyTheme(prefersDark ? 'dark' : 'light');
    }
})();

document.addEventListener('DOMContentLoaded', function () {
    const btn = document.getElementById('theme-toggle-btn');
    if (btn) {
        btn.addEventListener('click', toggleTheme);
    }

    // Loading state on the analyze form (nice UX touch --
    // pipeline processing takes a moment, this reassures the user)
    const form = document.getElementById('analyze-form');
    if (form) {
        form.addEventListener('submit', function () {
            const btn = form.querySelector('button[type="submit"]');
            if (btn) {
                btn.disabled = true;
                btn.textContent = 'Analyzing...';
            }
        });
    }
});
