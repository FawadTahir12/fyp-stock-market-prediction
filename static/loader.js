var loader = document.getElementById('loader'),
    load = document.getElementById('loading'),
    myTime,
    newTime = 0;
const form = document.getElementById('myform');
var    oloading = document.getElementById('loader');

form.addEventListener('submit', (e) => {
    e.preventDefault();
    // form.classList.toggle('fade-out');
    // spinner.classList.toggle('show');
        oloading.innerHTML =`
    <div class="loader-center" id="loading">
      <div class="loader-list">
          <div class="one loader-top"></div>
          <div class="two loader-top"></div>
          <div class="three loader-top"></div>
          <div class="four loader-top"></div>
          <div class="loader" id="loader">0%</div>
          <div class="one loader-bottom"></div>
          <div class="two loader-bottom"></div>
          <div class="three loader-bottom"></div>
          <div class="four loader-bottom"></div>
      </div>
    </div> `;
  });

function loading() {
    'use strict';

    newTime = newTime + 1;

    if (newTime > 100) {
        newTime = 0;
        // load.style.transition = '1s all';
        // load.style.opacity = '0';
        // clearInterval(myTime);
    } else {
        loader.textContent = newTime + "%";
    }
}

myTime = setInterval(loading, 500);

function onloading(){

    console.log('hello');
    
}

