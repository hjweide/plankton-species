var DELAY = 700, clicks = 0, timer = null;
$(document).ready(function() {
  $(function(){
    // because the elements are dynamically added:
    // http://stackoverflow.com/questions/203198/event-binding-on-dynamically-created-elements
    $('#selectable').on('click', 'img.lazy', function(e) {
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
                  $('#overlay_image_filepath').html('Filepath: ' + result['image_filepath']);
                  $('#overlay_image_date_collected').html('Collected on: ' + result['image_date_collected']);
                  $('#overlay_image_species').html('Species: ' + result['species_name']);
                  $('#overlay_image_added').html('Added by: ' + result['username_added'] + ' on ' + result['image_date_added']);
                  $('#overlay_image_annotated').html('Annotated by: ' + result['username_annotated'] + ' by ' + result['image_date_annotated']);
                  $('#overlay_image_dimensions').html('Height x Width: ' + result['image_height'] + ' x ' + result['image_width']);

                  // configure the html attributes of the overlay image
                  // want the full resolution image now...
                  img.attr('width', result['image_width']);
                  img.attr('height', result['image_height']);
                  img.attr('src', result['image_filepath'] + '?thumbnail=False');
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
    $('#overlayImage').attr('height', '');
    $('#overlayImage').attr('width', '');
    $('#overlay').hide();
    $('#overlayContent').hide();
  });
});
