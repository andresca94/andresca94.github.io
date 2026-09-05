---
title: "Data & AI Engineer"
---

<div class="portfolio-shell">
  <div class="portfolio-controls" data-portfolio-controls></div>
  <div class="portfolio-subfilters" data-portfolio-subfilters></div>

  <div class="portfolio-context">
    <div>
      <p class="portfolio-context-title" data-portfolio-context-title></p>
      <p class="portfolio-context-copy" data-portfolio-context-copy></p>
    </div>
    <p class="portfolio-count" data-portfolio-count></p>
  </div>

  <div class="projects-grid" data-projects-grid></div>

  <div class="portfolio-closing">
    <p>
      I am most energized by work that turns messy operational data into dependable products: pipelines that hold up in production, analytics that make decisions clearer, and AI systems grounded in real workflows. Across data, automation, research, and product engineering, I build tools that people can use, trust, and grow with.
    </p>
  </div>
</div>

{% if site.github.build_revision %}
  {% assign portfolio_asset_version = site.github.build_revision %}
{% else %}
  {% assign portfolio_asset_version = site.time | date: "%s" %}
{% endif %}

<script src="{{ '/assets/js/portfolio.js' | relative_url }}?v={{ portfolio_asset_version }}"></script>
