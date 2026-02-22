(() => {
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const navbar = document.getElementById("lp-navbar");
    const navLinks = Array.from(document.querySelectorAll(".lp-nav-link"));
    const sections = navLinks
        .map((link) => document.querySelector(link.getAttribute("href")))
        .filter(Boolean);

    function handleNavbarState() {
        if (!navbar) return;
        if (window.scrollY > 16) {
            navbar.classList.add("shrink");
        } else {
            navbar.classList.remove("shrink");
        }
    }

    function setActiveLink(id) {
        navLinks.forEach((link) => {
            const isActive = link.getAttribute("href") === `#${id}`;
            link.classList.toggle("active", isActive);
        });
    }

    if (sections.length) {
        // Keeps nav state aligned with currently visible section.
        const activeObserver = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        setActiveLink(entry.target.id);
                    }
                });
            },
            { threshold: 0.45 }
        );
        sections.forEach((section) => activeObserver.observe(section));
    }

    window.addEventListener("scroll", handleNavbarState, { passive: true });
    handleNavbarState();

    const revealEls = document.querySelectorAll(".lp-observe");
    if (revealEls.length) {
        if (reducedMotion) {
            revealEls.forEach((el) => el.classList.add("in-view"));
        } else {
            // Staggered in-view reveal; runs once per block for better performance.
            const revealObserver = new IntersectionObserver(
                (entries, observer) => {
                    entries.forEach((entry) => {
                        if (entry.isIntersecting) {
                            entry.target.classList.add("in-view");
                            observer.unobserve(entry.target);
                        }
                    });
                },
                { threshold: 0.2 }
            );
            revealEls.forEach((el) => revealObserver.observe(el));
        }
    }

    const rippleTargets = document.querySelectorAll(".lp-ripple");
    rippleTargets.forEach((target) => {
        target.addEventListener("click", (event) => {
            if (reducedMotion) return;
            const rect = target.getBoundingClientRect();
            const dot = document.createElement("span");
            dot.className = "lp-ripple-dot";
            dot.style.width = dot.style.height = `${Math.max(rect.width, rect.height) * 0.45}px`;
            dot.style.left = `${event.clientX - rect.left}px`;
            dot.style.top = `${event.clientY - rect.top}px`;
            target.appendChild(dot);
            dot.addEventListener("animationend", () => dot.remove(), { once: true });
        });
    });

    const magneticButtons = document.querySelectorAll(".lp-btn-magnetic");
    magneticButtons.forEach((btn) => {
        if (reducedMotion) return;
        btn.addEventListener("mousemove", (event) => {
            const rect = btn.getBoundingClientRect();
            const x = event.clientX - rect.left - rect.width / 2;
            const y = event.clientY - rect.top - rect.height / 2;
            btn.style.transform = `translate(${x * 0.08}px, ${y * 0.08}px)`;
        });
        btn.addEventListener("mouseleave", () => {
            btn.style.transform = "translate(0, 0)";
        });
    });

    const timeline = document.getElementById("lp-timeline");
    const timelineProgress = document.getElementById("lp-timeline-progress");
    const timelineSteps = Array.from(document.querySelectorAll(".lp-step"));

    function updateTimeline() {
        if (!timeline || !timelineProgress) return;
        const rect = timeline.getBoundingClientRect();
        const viewHeight = window.innerHeight;
        const total = rect.height + viewHeight * 0.2;
        const covered = Math.min(Math.max(viewHeight * 0.72 - rect.top, 0), total);
        const percent = Math.max(0, Math.min((covered / total) * 100, 100));
        timelineProgress.style.height = `${percent}%`;

        timelineSteps.forEach((step) => {
            const stepRect = step.getBoundingClientRect();
            const isActive = stepRect.top < viewHeight * 0.66 && stepRect.bottom > viewHeight * 0.28;
            step.classList.toggle("active", isActive);
        });
    }

    window.addEventListener("scroll", updateTimeline, { passive: true });
    window.addEventListener("resize", updateTimeline);
    updateTimeline();

    const moodInput = document.getElementById("lpMood");
    const moodLabel = document.getElementById("lpMoodLabel");
    if (moodInput && moodLabel) {
        const moodMap = {
            "1": "Struggling",
            "2": "Low",
            "3": "Balanced",
            "4": "Better",
            "5": "Calm"
        };
        const applyMood = () => {
            moodLabel.textContent = moodMap[moodInput.value] || "Balanced";
        };
        moodInput.addEventListener("input", applyMood);
        applyMood();
    }

    const breathCircle = document.getElementById("lpBreathCircle");
    const resetBreathBtn = document.getElementById("lpResetBreath");
    if (breathCircle && resetBreathBtn) {
        /* Resets breathing animation in a controlled way for user-guided loops */
        resetBreathBtn.addEventListener("click", () => {
            breathCircle.classList.add("resetting");
            // Force reflow to restart keyframes
            void breathCircle.offsetWidth;
            breathCircle.classList.remove("resetting");
        });
    }
})();
