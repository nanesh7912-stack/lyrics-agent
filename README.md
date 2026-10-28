# AI Lyrics Hero website

This responsive introduction site uses plain HTML and CSS. It has no build step and works in modern phone, tablet, and desktop browsers.

## Pages

- Home introduces the project and its approach to lyric videos.
- About explains the project's purpose.
- Features describes its lyric timing and video workflow.
- Contact includes an email form and direct email link.

This site introduces the project. It does not accept audio uploads or render videos in the browser.

## Contact form

The form uses [FormSubmit](https://formsubmit.co/) to forward messages to nanesh7912@gmail.com. On the first submission after publishing, FormSubmit sends a one-time activation email to that inbox. Confirm it before expecting form messages. The recipient email is visible in the page source. FormSubmit's processing and privacy terms apply. Do not submit a test message unless you intend to send a real message.

## Publish with GitHub Pages

The site uses six files: index.html, about.html, features.html, contact.html, site.css, and favicon.svg. Include this README.md in the repository for project and publishing instructions. The full AI Lyrics Hero project can contain model files, audio, and test materials. To avoid publishing those files, create a dedicated public website repository and upload only these seven site and documentation files.

### Upload from a browser

1. Create a new public GitHub repository for the website.
2. Choose Add file, then Upload files. Upload the six site files and this README.md so index.html is at the repository root.
3. Commit the files to the main branch.
4. In Settings → Pages, choose Deploy from a branch, select main and /(root), then save.
5. Wait for deployment. GitHub shows the public website address in Settings → Pages.

### Publish with Git

1. Create a public empty GitHub repository for the website.
2. Copy the six site files and README.md into an empty local folder.
3. Open PowerShell in that folder and run the following, replacing YOUR-USERNAME with your account name:

~~~
git init
git add .
git commit -m "Add AI Lyrics Hero website"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ai-lyrics-hero-site.git
git push -u origin main
~~~

4. In Settings → Pages, choose Deploy from a branch, select main and /(root), then save. Open the website address shown there after deployment.

See the [GitHub Pages publishing guide](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) for current settings.

## Device, browser, and accessibility checks

Open the published link in current Chrome, Edge, Firefox, and Safari. Check it at phone, tablet, and desktop widths. Follow each navigation link, resize the page, and check that no content is clipped.

For keyboard access, press Tab from the top of each page and verify the skip link, navigation, form fields, and buttons have a visible focus outline. Check at 200% zoom, try a screen reader such as VoiceOver, NVDA, or TalkBack, and review colors with an accessibility checker. The site includes semantic landmarks, linked form labels, responsive layouts, visible focus states, and reduced-motion support. These are accessibility practices, not a claim of formal WCAG certification.

## Preview and form activation

Open index.html in a browser to preview the pages. The contact form requires the published HTTPS website and the one-time FormSubmit email activation. Opening a page as a local file cannot deliver form submissions.
