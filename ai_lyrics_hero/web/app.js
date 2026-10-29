(() => {
  const showStartupError = (error) => {
    const root = document.querySelector('#app');
    if (!root) return;
    root.removeAttribute('v-cloak');
    root.innerHTML = '<main role="alert" style="max-width:560px;margin:12vh auto;padding:28px;color:#eef3ff;font:16px/1.6 system-ui,sans-serif"><h1 style="font-size:24px">The browser studio could not start</h1><p>Reload the page. If this keeps happening, restart the local video app and try again.</p></main>';
    if (error) {
      const detail = document.createElement('p');
      detail.style.color = '#ffb8c5';
      detail.textContent = error.message || String(error);
      root.querySelector('main')?.append(detail);
    }
  };

  try {
    if (!window.Vue || typeof window.Vue.createApp !== 'function') {
      throw new Error('The Vue runtime did not load.');
    }
    const { createApp } = window.Vue;
    const app = createApp({
    data() {
      let savedDefaults = {};
      try { savedDefaults = JSON.parse(localStorage.getItem('lyricsHeroDefaults') || '{}'); } catch (_) { savedDefaults = {}; }
      return {
        view: thisViewFromPath(),
        mobileMenuOpen: false,
        step: 1,
        csrfToken: '',
        contactEmail: 'nanesh7912@gmail.com',
        audioFile: null,
        audioUrl: '',
        isDragging: false,
        lyricsMode: 'auto',
        lyricsFormat: 'plain',
        lyricsText: '',
        submitting: false,
        activeJob: null,
        pollTimer: null,
        projects: [],
        jobs: [],
        projectId: null,
        toast: '',
        toastType: 'success',
        toastTimer: null,
        settingsSaved: false,
        contact: { name: '', email: '', subject: '', message: '' },
        defaults: {
          duration: Number(savedDefaults.duration) || 40,
          language: savedDefaults.language || '',
          romanize: Boolean(savedDefaults.romanize),
        },
        options: {
          title: 'My lyric video', trackTitle: '', artist: '', lyricsLookup: true,
          start: 0, autoSelect: true, duration: Number(savedDefaults.duration) || 40,
          size: '1080x1920', fps: 30, style: 'scene-hero', language: savedDefaults.language || '',
          romanize: Boolean(savedDefaults.romanize),
        },
        steps: ['Input', 'Timing', 'Style', 'Export'],
        primaryNav: [
          { id: 'create', label: 'Create video', icon: '✦', href: '/' },
          { id: 'projects', label: 'Projects', icon: '▣', href: '/projects' },
          { id: 'templates', label: 'Templates', icon: '◫', href: '/templates' },
          { id: 'settings', label: 'Settings', icon: '⚙', href: '/settings' },
        ],
        infoNav: [
          { id: 'help', label: 'Help center', icon: '?', href: '/help' },
          { id: 'about', label: 'About', icon: '◎', href: '/about' },
          { id: 'services', label: 'Services', icon: '◇', href: '/services' },
          { id: 'contact', label: 'Contact', icon: '✉', href: '/contact' },
        ],
        styles: [
          { id: 'scene-hero', title: 'Scene Hero', description: 'Dreamy moving scenes with expressive hero words.' },
          { id: 'scene', title: 'Night Scenes', description: 'Atmospheric backgrounds with clear lyric captions.' },
          { id: 'hero', title: 'Hero Words', description: 'Bold typography and a pulsing music visualizer.' },
          { id: 'classic', title: 'Classic', description: 'Bright geometric backgrounds and glowing lyrics.' },
        ],
      };
    },
    computed: {
      selectedStyle() { return this.styles.find((item) => item.id === this.options.style) || this.styles[0]; },
      pageTitle() {
        const titles = { create: 'Create video', projects: 'Projects', templates: 'Templates', settings: 'Settings',
          help: 'Help center', about: 'About', services: 'Services', contact: 'Contact' };
        return titles[this.view] || 'Create video';
      },
    },
    async mounted() {
      window.addEventListener('popstate', this.onPopState);
      try {
        const info = await this.api('/api/session');
        this.csrfToken = info.csrf_token;
        this.contactEmail = info.contact_email || this.contactEmail;
        await Promise.all([this.refreshProjects(), this.refreshJobs()]);
      } catch (error) { this.showToast(error.message || 'Could not connect to the video service.', 'error'); }
    },
    beforeUnmount() {
      window.removeEventListener('popstate', this.onPopState);
      if (this.pollTimer) clearTimeout(this.pollTimer);
      if (this.toastTimer) clearTimeout(this.toastTimer);
      if (this.audioUrl) URL.revokeObjectURL(this.audioUrl);
    },
    methods: {
      async api(url, options = {}) {
        const headers = new Headers(options.headers || {});
        if (options.method && options.method !== 'GET') headers.set('X-CSRF-Token', this.csrfToken);
        if (options.body && !(options.body instanceof FormData)) headers.set('Content-Type', 'application/json');
        const response = await fetch(url, { credentials: 'same-origin', ...options, headers });
        if (response.status === 204) return null;
        const type = response.headers.get('content-type') || '';
        const payload = type.includes('application/json') ? await response.json() : null;
        if (!response.ok) throw new Error(payload?.error || `Request failed (${response.status}).`);
        return payload;
      },
      navigate(page) {
        const paths = { create: '/', projects: '/projects', templates: '/templates', settings: '/settings', help: '/help', about: '/about', services: '/services', contact: '/contact' };
        this.view = paths[page] ? page : 'create';
        this.mobileMenuOpen = false;
        window.history.pushState({ page: this.view }, '', paths[this.view]);
        window.scrollTo({ top: 0, behavior: 'smooth' });
        if (page === 'projects') Promise.all([this.refreshProjects(), this.refreshJobs()]).catch((error) => this.showToast(error.message, 'error'));
      },
      onPopState() { this.view = thisViewFromPath(); this.mobileMenuOpen = false; },
      async refreshProjects() {
        const data = await this.api('/api/projects');
        this.projects = data.projects;
      },
      async refreshJobs() {
        const data = await this.api('/api/jobs');
        this.jobs = data.jobs;
      },
      onAudioChange(event) {
        const file = event.target.files && event.target.files[0];
        if (file) this.useAudioFile(file);
        event.target.value = '';
      },
      onDrop(event) {
        this.isDragging = false;
        const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
        if (file) this.useAudioFile(file);
      },
      useAudioFile(file) {
        if (file.size > 100 * 1024 * 1024) { this.showToast('Audio files must be smaller than 100 MB.', 'error'); return; }
        if (file.type && !file.type.startsWith('audio/') && !/\.(mp3|wav|m4a|aac|flac|ogg|opus|wma|mp4)$/i.test(file.name)) {
          this.showToast('Choose a supported audio file, such as MP3 or WAV.', 'error'); return;
        }
        if (this.audioUrl) URL.revokeObjectURL(this.audioUrl);
        this.audioFile = file;
        this.audioUrl = URL.createObjectURL(file);
        const stem = file.name.replace(/\.[^.]+$/, '');
        const parts = stem.split(/\s+-\s+/);
        this.options.trackTitle = parts[0] || stem;
        this.options.artist = parts.length > 1 ? parts.slice(1).join(' - ') : '';
      },
      clearAudio() {
        if (this.audioUrl) URL.revokeObjectURL(this.audioUrl);
        this.audioFile = null;
        this.audioUrl = '';
        this.options.trackTitle = '';
        this.options.artist = '';
      },
      formatSize(bytes) {
        if (!Number.isFinite(bytes)) return '';
        return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
      },
      loadExample() {
        this.lyricsMode = 'custom';
        this.lyricsFormat = 'plain';
        this.lyricsText = 'Are re are yeh kya hua\nMaine na yeh jaana\nAre re ban jaaye na\nKahin koi afsana';
      },
      continueStep() {
        if (this.step === 1 && !this.audioFile) { this.showToast('Choose a song before you continue.', 'error'); return; }
        if (this.step === 2 && (!Number.isFinite(Number(this.options.start)) || Number(this.options.start) < 0 || !Number.isFinite(Number(this.options.duration)) || Number(this.options.duration) < 2 || Number(this.options.duration) > 600)) {
          this.showToast('Use a start time from 0 and a video length from 2 to 600 seconds.', 'error'); return;
        }
        this.step = Math.min(4, this.step + 1);
        window.scrollTo({ top: 0, behavior: 'smooth' });
      },
      async createVideo() {
        if (!this.audioFile || this.submitting) return;
        this.submitting = true;
        this.activeJob = null;
        const [width, height] = this.options.size.split('x').map(Number);
        const form = new FormData();
        form.append('audio', this.audioFile, this.audioFile.name);
        form.append('title', this.options.title || 'My lyric video');
        form.append('style', this.options.style);
        form.append('language', this.lyricsMode === 'auto' ? this.options.language : '');
        form.append('romanize', String(this.options.romanize));
        form.append('start', String(this.options.start));
        form.append('auto_select', String(this.options.autoSelect));
        form.append('duration', String(this.options.duration));
        form.append('size', `${width}x${height}`);
        form.append('fps', String(this.options.fps));
        form.append('model', 'medium');
        form.append('track_title', this.options.trackTitle || '');
        form.append('artist', this.options.artist || '');
        form.append('lyrics_lookup', String(this.options.lyricsLookup));
        if (this.lyricsMode === 'custom') form.append('lyrics_text', this.lyricsText);
        try {
          const result = await this.api('/api/jobs', { method: 'POST', body: form });
          this.activeJob = result.job;
          this.pollJob();
          this.showToast('Your video is in the render queue.');
        } catch (error) { this.showToast(error.message, 'error'); }
        finally { this.submitting = false; }
      },
      async pollJob() {
        if (!this.activeJob) return;
        try {
          const data = await this.api(`/api/jobs/${this.activeJob.id}`);
          this.activeJob = data.job;
          if (this.activeJob.status === 'completed' || this.activeJob.status === 'failed') {
            if (this.pollTimer) clearTimeout(this.pollTimer);
            this.pollTimer = null;
            await this.refreshJobs();
          } else {
            this.pollTimer = setTimeout(() => this.pollJob(), 1400);
          }
        } catch (error) {
          this.showToast(error.message || 'Lost connection while checking the video.', 'error');
          this.pollTimer = setTimeout(() => this.pollJob(), 3000);
        }
      },
      styleName(style) { return this.styles.find((item) => item.id === style)?.title || 'Scene Hero'; },
      async saveProject() {
        const settings = { ...this.options };
        settings.lyricsMode = this.lyricsMode;
        settings.lyricsText = this.lyricsText;
        settings.lyricsFormat = this.lyricsFormat;
        const isNew = !this.projectId;
        try {
          let result;
          if (this.projectId) {
            result = await this.api(`/api/projects/${this.projectId}`, { method: 'PUT', body: JSON.stringify({ title: this.options.title, settings }) });
          } else {
            result = await this.api('/api/projects', { method: 'POST', body: JSON.stringify({ title: this.options.title, settings }) });
            this.projectId = result.project.id;
          }
          await this.refreshProjects();
          this.showToast(isNew ? 'Project saved.' : 'Project settings saved.');
        } catch (error) { this.showToast(error.message, 'error'); }
      },
      loadProject(project) {
        const settings = project.settings || {};
        this.options = { ...this.options, ...settings };
        this.lyricsMode = settings.lyricsMode || 'auto';
        this.lyricsText = settings.lyricsText || '';
        this.lyricsFormat = settings.lyricsFormat || 'plain';
        this.projectId = project.id;
        this.activeJob = null;
        this.step = 1;
        this.navigate('create');
        this.showToast('Project settings loaded. Add the song to continue.');
      },
      async deleteProject(project) {
        if (!window.confirm(`Delete the saved project “${project.title}”?`)) return;
        try { await this.api(`/api/projects/${project.id}`, { method: 'DELETE' }); await this.refreshProjects(); this.showToast('Project deleted.'); }
        catch (error) { this.showToast(error.message, 'error'); }
      },
      async deleteJob(job) {
        if (!window.confirm(`Delete “${job.title}” and its rendered video? This cannot be undone.`)) return;
        try { await this.api(`/api/jobs/${job.id}`, { method: 'DELETE' }); await this.refreshJobs(); this.showToast('Video deleted.'); }
        catch (error) { this.showToast(error.message, 'error'); }
      },
      chooseTemplate(style) {
        this.options.style = style;
        this.navigate('create');
        this.step = 3;
      },
      newVideo() {
        if (this.pollTimer) clearTimeout(this.pollTimer);
        this.pollTimer = null;
        this.activeJob = null;
        this.projectId = null;
        this.step = 1;
        this.lyricsMode = 'auto';
        this.lyricsText = '';
        this.options.title = 'My lyric video';
        this.options.start = 0;
        this.options.autoSelect = true;
        this.options.duration = this.defaults.duration;
        this.options.language = this.defaults.language;
        this.options.romanize = this.defaults.romanize;
        this.navigate('create');
      },
      saveDefaults() {
        const duration = Math.max(2, Math.min(600, Number(this.defaults.duration) || 40));
        this.defaults.duration = duration;
        localStorage.setItem('lyricsHeroDefaults', JSON.stringify(this.defaults));
        this.settingsSaved = true;
        setTimeout(() => { this.settingsSaved = false; }, 2400);
        this.showToast('Your preferences were saved on this device.');
      },
      sendContact() {
        const subject = encodeURIComponent(this.contact.subject.trim());
        const body = encodeURIComponent(`From: ${this.contact.name.trim()}\nReply to: ${this.contact.email.trim()}\n\n${this.contact.message.trim()}`);
        window.location.href = `mailto:${encodeURIComponent(this.contactEmail)}?subject=${subject}&body=${body}`;
      },
      formatDate(value) {
        if (!value) return '';
        const date = new Date(value);
        return Number.isNaN(date.valueOf()) ? '' : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' }).format(date);
      },
      statusLabel(status) { return ({ queued: 'Queued', running: 'Rendering', completed: 'Ready', failed: 'Failed' })[status] || status; },
      showToast(message, type = 'success') {
        this.toast = message;
        this.toastType = type;
        if (this.toastTimer) clearTimeout(this.toastTimer);
        this.toastTimer = setTimeout(() => { this.toast = ''; }, 4200);
      },
    },
    });
    app.config.errorHandler = (error) => {
      console.error(error);
      showStartupError(error);
    };
    app.mount('#app');
  } catch (error) {
    console.error(error);
    showStartupError(error);
  }

  function thisViewFromPath() {
    const path = window.location.pathname.replace(/\/$/, '') || '/';
    const pages = { '/': 'create', '/projects': 'projects', '/templates': 'templates', '/settings': 'settings', '/help': 'help', '/about': 'about', '/services': 'services', '/contact': 'contact' };
    return pages[path] || 'create';
  }
})();
