(() => {
  const programs = [
    { value: "ai-edge-system", label: "AI Edge System" },
    { value: "hec", label: "HEC" },
  ];
  const genericProgram = { value: "generic", label: "Generic" };
  const validPrograms = new Set(programs.map((program) => program.value));
  // Pages (under getting-started/) that render program panels and must keep the
  // selected program when cross-linked from one another, e.g. Quick Start <-> Requirements.
  const programPageSlugs = new Set(["quick-start", "requirements"]);

  const isProgramPageUrl = (url) => {
    const segments = url.pathname.replace(/\/index\.html$/, "").split("/").filter(Boolean);
    const last = segments[segments.length - 1];
    return segments[segments.length - 2] === "getting-started" && programPageSlugs.has(last);
  };

  const updateMenuState = (menu) => {
    const trigger = menu.querySelector(".program-menu__trigger");
    if (trigger) {
      trigger.setAttribute("aria-expanded", String(menu.matches(":hover, :focus-within")));
    }
  };

  const closeMenuUntilReentry = (menu) => {
    menu.classList.add("program-menu--closed");
    menu.querySelector(".program-menu__trigger")?.setAttribute("aria-expanded", "false");

    // collapsing the list removes it from under the cursor, which the browser treats as a real
    // mouseleave even though the pointer hasn't moved - track actual pointer position instead
    const isPointerOutside = (event) => {
      const rect = menu.getBoundingClientRect();
      return (
        event.clientX < rect.left || event.clientX > rect.right ||
        event.clientY < rect.top || event.clientY > rect.bottom
      );
    };

    const cleanup = () => {
      menu.classList.remove("program-menu--closed");
      document.removeEventListener("mousemove", handleMouseMove);
      menu.removeEventListener("focusout", handleFocusOut);
    };

    const handleMouseMove = (event) => {
      if (isPointerOutside(event)) cleanup();
    };

    const handleFocusOut = () => {
      // focus hasn't moved to its new target yet when focusout fires; check on the next tick
      setTimeout(() => {
        if (!menu.matches(":focus-within")) cleanup();
      }, 0);
    };

    document.addEventListener("mousemove", handleMouseMove);
    menu.addEventListener("focusout", handleFocusOut);
  };

  const createMenu = (hasGenericPanel) => {
    const header = document.querySelector(".md-header__inner");
    const title = header?.querySelector(".md-header__title");
    if (!title) return;
    const availablePrograms = hasGenericPanel ? [genericProgram, ...programs] : programs;

    let element = document.querySelector(".program-menu");
    if (!element) {
      element = document.createElement("div");
      element.className = "program-menu";
      element.innerHTML = `
        <button class="program-menu__trigger" type="button" aria-haspopup="true" aria-expanded="false">
          <span class="program-menu__current"></span>
        </button>
        <div class="program-menu__list" role="menu" aria-label="Select a program"></div>
      `;
      title.insertAdjacentElement("afterend", element);
      ["mouseenter", "mouseleave", "focusin", "focusout"].forEach((eventName) => {
        element.addEventListener(eventName, () => updateMenuState(element));
      });
    }

    element.querySelector(".program-menu__list").innerHTML = availablePrograms.map((program) => `
      <button class="program-menu__option" type="button" role="menuitemradio" aria-checked="false" data-program-choice="${program.value}">
        ${program.label}
      </button>
    `).join("");
  };

  const getSelectedProgram = (hasGenericPanel) => {
    const requestedProgram = new URLSearchParams(window.location.search).get("program");
    if (hasGenericPanel && requestedProgram === genericProgram.value) return genericProgram.value;
    if (validPrograms.has(requestedProgram)) return requestedProgram;
    if (hasGenericPanel) return genericProgram.value;
    return "ai-edge-system";
  };

  const showProgram = (program) => {
    if (program !== genericProgram.value && !validPrograms.has(program)) return;
    document.querySelectorAll("[data-program-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.programPanel !== (program === genericProgram.value ? "generic" : program);
    });

    const selected = [genericProgram, ...programs].find((entry) => entry.value === program);
    document.querySelectorAll("[data-selected-program]").forEach((element) => {
      if (selected) element.textContent = selected.label;
    });
    const menu = document.querySelector(".program-menu");
    if (menu) {
      const trigger = menu.querySelector(".program-menu__trigger");
      const label = selected?.label ?? "Select a program";
      trigger.setAttribute("aria-label", selected ? `Select a program. Current program: ${label}` : label);
      trigger.querySelector(".program-menu__current").textContent = label;
      menu.querySelectorAll("[data-program-choice]").forEach((option) => {
        option.setAttribute("aria-checked", String(option.dataset.programChoice === program));
      });
    }

    if (program) document.querySelectorAll(".md-nav--secondary a[href*='#']").forEach((link) => {
      const linkUrl = new URL(link.href);
      linkUrl.searchParams.set("program", program);
      link.href = linkUrl;
      const targetId = decodeURIComponent(linkUrl.hash.slice(1));
      const heading = document.getElementById(targetId);
      const panel = heading?.closest("[data-program-panel]");
      const item = link.closest("li");
      if (item && panel) item.hidden = panel.dataset.programPanel !== program;
    });

    // Carry the selected program across cross-page links between program-aware pages
    // (e.g. the "Requirements" link on Quick Start and the "Quick Start" link on Requirements)
    // so the destination page opens with the same program already selected.
    if (program) document.querySelectorAll(".md-content a[href]").forEach((link) => {
      let linkUrl;
      try {
        linkUrl = new URL(link.href);
      } catch {
        return;
      }
      if (linkUrl.origin !== window.location.origin || !isProgramPageUrl(linkUrl)) return;
      linkUrl.searchParams.set("program", program);
      link.href = linkUrl;
    });

    const currentUrl = new URL(window.location.href);
    currentUrl.searchParams.set("program", program);
    window.history.replaceState(window.history.state, "", currentUrl);
  };

  const initialize = () => {
    const panels = document.querySelectorAll("[data-program-panel]");
    if (!panels.length) {
      document.querySelector(".program-menu")?.remove();
      return;
    }
    const hasGenericPanel = Array.from(panels).some((panel) => panel.dataset.programPanel === "generic");
    createMenu(hasGenericPanel);
    showProgram(getSelectedProgram(hasGenericPanel));
  };

  document.addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target : null;
    const option = target?.closest("[data-program-choice]");

    if (option) {
      showProgram(option.dataset.programChoice);
      const menu = option.closest(".program-menu");
      if (menu) closeMenuUntilReentry(menu);
      // a clicked button keeps keyboard focus, which would keep the list open via
      // :focus-within even after the mouse leaves - release it so only hover matters
      option.blur();
      return;
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      const menu = document.querySelector(".program-menu");
      const trigger = menu?.querySelector(".program-menu__trigger");
      if (menu?.matches(":focus-within")) trigger?.blur();
      if (menu) updateMenuState(menu);
    }
  });

  if (window.document$) {
    window.document$.subscribe(initialize);
  } else {
    initialize();
  }
})();