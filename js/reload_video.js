
function ajax(URL, successCallback, errorCallback){
  $.ajax({
    url: URL,
    type: "GET",
    dataType: "text",
    scriptCharset: "utf-8",
    success: successCallback,
    error: errorCallback
  }); // $.ajax
} // function ajax

var videos = {
  v09: document.getElementById("vid09"),
  v10: document.getElementById("vid10"),
  v11: document.getElementById("vid11"),
  c09: document.getElementById("cam09"),
  c10: document.getElementById("cam10"),
  c11: document.getElementById("cam11"),
  m09: document.getElementById("mask09"),
  m10: document.getElementById("mask10"),
  m11: document.getElementById("mask11"),
};

function setAndPlay(){
  videos.v09.currentTime = 0;
  videos.v10.currentTime = 0;
  videos.v11.currentTime = 0;
  videos.c09.currentTime = 0;
  videos.c10.currentTime = 0;
  videos.c11.currentTime = 0;
  videos.m09.currentTime = 0;
  videos.m10.currentTime = 0;
  videos.m11.currentTime = 0;
  videos.v09.play();
  videos.v10.play();
  videos.v11.play();
  videos.c09.play();
  videos.c10.play();
  videos.c11.play();
  videos.m09.play();
  videos.m10.play();
  videos.m11.play();
};


document.getElementById('playVid').onclick = function (){
  setAndPlay();
};

$(document).ready(function(){
  setAndPlay();
});

