var DELAY = 700, clicks = 0, timer = null;
function update_overlay(image_id) {
  var img = $('#overlayImage');
  // query the database for the chosen image's info
  $.ajax({
    url: '/overlay',
    data: jQuery.param({'image_id': image_id}),
    type: 'POST',
    success: function(response) {
      console.log('success: ' + response);
      result = JSON.parse(response);

      // populate the overlay image's info
      $('#overlay_image_filepath').html('Filepath: ' + result['image_filepath']);
      $('#overlay_image_date_collected').html('Collected on: ' + result['image_date_collected']);
      $('#overlay_image_family').html(
        'Family: ' + result['family_name'] +
        ' (confusable is ' + result['family_confusable'] +
        ' annotated by ' + result['username_family_annotated'] +
        ' on ' + result['image_date_family_annotated'] + ')');
      $('#overlay_image_genus').html(
        'Genus: ' + result['genus_name'] +
        ' (confusable is ' + result['genus_confusable'] +
        ' annotated by ' + result['username_genus_annotated'] +
        ' on ' + result['image_date_genus_annotated'] + ')');
      $('#overlay_image_species').html(
        'Species: ' + result['species_name'] +
        ' (confusable is ' + result['species_confusable'] +
        ' annotated by ' + result['username_species_annotated'] +
        ' on ' + result['image_date_species_annotated'] + ')');
      $('#overlay_image_added').html('Added by: ' + result['username_added'] + ' on ' + result['image_date_added']);
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
}
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
              update_overlay(e.target.id);
          }
      })
      .on("dblclick", function(e){
          e.preventDefault();  //cancel system double-click event
      });
  });
});
$(document).on('keydown', function(e) {

  // left arrow key
  if (e.keyCode == 37) {
    // only move when the overlay is visible
    if ($('#overlay').is(':hidden'))
      return false;

    // find the currently selected image and its previousSibling
    var current_image = $('#selectable.ui-selectable').children('.ui-selected');
    var previous_image = current_image.prev();

    // need to update the jquery selectable
    current_image.removeClass('ui-selected');
    previous_image.addClass('ui-selected');

    // query the database for the new overlay
    var image_id_left = previous_image.children('img.lazy').attr('id');
    update_overlay(image_id_left);
  
    return false;
  }
  // right arrow key
  else if (e.keyCode == 39) {
    // only move when the overlay is visible
    if ($('#overlay').is(':hidden'))
      return false;

    // find the currently selected image and its nextSibling
    var current_image = $('#selectable.ui-selectable').children('.ui-selected');
    var next_image = current_image.next();

    // need to update the jquery selectable
    current_image.removeClass('ui-selected');
    next_image.addClass('ui-selected');

    // query the database for the new overlay
    var image_id_right = next_image.children('img.lazy').attr('id');
    update_overlay(image_id_right);
    return false;
  }
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
