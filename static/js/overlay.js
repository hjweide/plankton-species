var DELAY = 700, clicks = 0, timer = null;
$(document).ready(function() {
  $(function(){
      $("img.lazy").on("click", function(e){
          clicks++;  //count clicks
          if(clicks === 1) {
              timer = setTimeout(function() {
                  clicks = 0;             //after action performed, reset counter
                  e.stopPropagation();
              }, DELAY);
          } else {
              clearTimeout(timer);    //prevent single-click action
              var img = $('#overlayImage');

              // query the database for the chosen image's info
              $.ajax({
                url: '/overlay',
                data: jQuery.param({'image_id': e.target.id}),
                type: 'POST',
                success: function(response) {
                  console.log('success: ' + response);
                  result = JSON.parse(response);

                  // populate the overlay image's info
                  document.getElementById('overlay_image_filepath').innerHTML = 'Filepath: ' + result['image_filepath'];
                  document.getElementById('overlay_image_species').innerHTML = 'Species: ' + result['species_name'];
                  document.getElementById('overlay_image_user').innerHTML = 'Added by: ' + result['username'];
                  document.getElementById('overlay_image_dimensions').innerHTML = 'Dimensions: ' + result['image_height'] + ' x ' + result['image_width'];

                  // configure the html attributes of the overlay image
                  // want the full resolution image now...
                  img.attr('src', result['image_filepath'] + '?thumbnail=False');
                  img.attr('width', result['image_width']);
                  img.attr('height', result['image_height']);
                },
                error: function(error) {
                  console.log('error: ' + error);
                }
              });

              var content = $('#overlayContent');
              // center the image on the center of the viewport
              content.css({
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%)",
              });
              $('#overlay').show();
              $('#overlayContent').show();
              clicks = 0;             //after action performed, reset counter
          }
      })
      .on("dblclick", function(e){
          e.preventDefault();  //cancel system double-click event
      });
  });
});
$(document).ready(function() {
    $('#overlayContent').click(function() {
    $('#overlayImage').attr('src', '');
    $('#overlay').hide();
    $('#overlayContent').hide();
  });
});
