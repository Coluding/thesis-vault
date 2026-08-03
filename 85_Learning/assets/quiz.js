/* Reusable quiz + free-recall components for the 85_Learning workspace.
 *
 * Markup contract:
 *
 *   <div class="quiz" data-answer="2">
 *     <p class="q">Question text</p>
 *     <div class="opt">Option A</div>
 *     <div class="opt">Option B</div>
 *     <div class="opt">Option C</div>
 *     <div class="fb" data-correct="Shown when right." data-wrong="Shown when wrong."></div>
 *   </div>
 *
 *   <div class="recall">
 *     <p class="q">Prompt</p>
 *     <textarea></textarea>
 *     <button>Show a model answer</button>
 *     <div class="model">…model answer…</div>
 *   </div>
 *
 * data-answer is the 0-indexed correct option. Feedback is revealed on first
 * click and the question locks — one attempt, so the retrieval is effortful.
 */
(function () {
  function initQuiz(root) {
    var answer = parseInt(root.getAttribute('data-answer'), 10);
    var opts = Array.prototype.slice.call(root.querySelectorAll('.opt'));
    var fb = root.querySelector('.fb');
    var done = false;

    opts.forEach(function (opt, i) {
      opt.setAttribute('role', 'button');
      opt.setAttribute('tabindex', '0');

      function choose() {
        if (done) return;
        done = true;
        opts.forEach(function (o, j) {
          o.classList.add('locked');
          o.removeAttribute('tabindex');
          if (j === answer) o.classList.add('correct');
        });
        if (i !== answer) opt.classList.add('wrong');
        if (fb) {
          var key = i === answer ? 'data-correct' : 'data-wrong';
          var txt = fb.getAttribute(key) || fb.getAttribute('data-correct') || '';
          fb.innerHTML = (i === answer ? '<b>Correct.</b> ' : '<b>Not quite.</b> ') + txt;
          fb.classList.add('show');
        }
      }

      opt.addEventListener('click', choose);
      opt.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(); }
      });
    });
  }

  function initRecall(root) {
    var btn = root.querySelector('button');
    var model = root.querySelector('.model');
    if (!btn || !model) return;
    btn.addEventListener('click', function () {
      model.classList.add('show');
      btn.disabled = true;
      btn.style.opacity = '.5';
      btn.textContent = 'Compare yours to this';
    });
  }

  function boot() {
    document.querySelectorAll('.quiz[data-answer]').forEach(initQuiz);
    document.querySelectorAll('.recall').forEach(initRecall);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
