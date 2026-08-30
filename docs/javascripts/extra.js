/* Reading progress bar — article ke scroll ke hisaab se upar patli line bharti hai.
   document$ Material ka observable hai; instant navigation ke baad bhi chalta rehta hai. */

document$.subscribe(function () {
  var bar = document.querySelector(".nt-progress");

  if (!bar) {
    bar = document.createElement("div");
    bar.className = "nt-progress";
    document.body.appendChild(bar);
  }

  var article = document.querySelector(".md-content__inner");

  function update() {
    if (!article) {
      bar.style.transform = "scaleX(0)";
      return;
    }

    var top = article.offsetTop;
    var scrollable = article.offsetHeight - window.innerHeight + top;
    var progress = scrollable > 0
      ? Math.min(Math.max((window.scrollY - top) / scrollable, 0), 1)
      : 0;

    bar.style.transform = "scaleX(" + progress + ")";
  }

  update();
  window.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update, { passive: true });
});
