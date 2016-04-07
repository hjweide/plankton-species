// Trigger action when the contexmenu is about to be shown
$(document).bind("contextmenu", function (event) {
    // Avoid the real one
    event.preventDefault();
    // Show contextmenu
    $("#family-menu").finish().toggle(100).
    // In the right position (the mouse)
    css({
        top: event.pageY + "px",
        left: event.pageX + "px",
    });
});
// If the document is clicked somewhere
$(document).bind("mousedown", function (e) {
    // If the clicked element is not the menu
    if (!$(e.target).parents("#family-menu").length > 0) {
        // Hide it
        $("#family-menu").hide(100);
    }
});
$(document).ready(function() {
  $("ul#family-menu li").hover(function () {
    $(this).children('ul').show();
  }, function() {
    $(this).children('ul').hide();
  });
});
$(document).ready(function() {
  $(function(){
    // trigger only when the direct descendant list item is clicked
    $('#species-menu > li').on('click', function(e) {
      var clicked = $(this);
      var species_id = clicked.attr('id');
      var species_name = clicked.attr('data-action');
      console.log("species_id: " + clicked.attr('id') + ", species_name: " + species_name);
      e.stopPropagation();
    });
    $('#genus-menu > li').on('click', function(e) {
      var clicked = $(this);
      var genus_id = clicked.attr('id');
      var genus_name = clicked.attr('data-action');
      console.log("genus_id: " + clicked.attr('id') + ", genus_name: " + genus_name);
      e.stopPropagation();
    });
    $('#family-menu > li').on('click', function(e) {
      var clicked = $(this);
      var family_id = clicked.attr('id');
      var family_name = clicked.attr('data-action');
      console.log("family_id: " + clicked.attr('id') + ", family_name: " + family_name);
      e.stopPropagation();
    });
  });
  //$("#species-menu li").click(function(event){
  //  event.stopPropagation();
  //  var clicked = $(this);
  //  var species_id = clicked.attr('id');
  //  var species = clicked.attr('data-action');
  //  console.log("species_id: " + species_id + ", species: " + species);
  //});
  //$("#genus-menu li").click(function(event){
  //  event.stopPropagation();
  //  var clicked = $(this);
  //  var genus_id = clicked.attr('id');
  //  var genus = clicked.attr('data-action');
  //  console.log("genus_id: " + genus_id + ", genus: " + genus);
  //});
  //// If the menu element is clicked
  //$("#family-menu li").click(function(event){
  //    // This is the triggered action name
  //    event.stopPropagation();
  //    var clicked = $(this);
  //    console.log(clicked);
  //    var family_id = clicked.attr('id');
  //    var family = clicked.attr('data-action');
  //    console.log("family_id: " + family_id + ", family: " + family);
  //    return;
  //    var clicked = $(this);
  //    var species = clicked.attr('data-action');
  //    var cls = clicked.attr('cls');
  //    var species_id = clicked.attr('id');
  //    console.log('species_id: ' + species_id + ', species: ' + species);
  //    var children = $('#selectable').children('.ui-selected');

  //    // hide it AFTER the action was triggered
  //    $("#family-menu").hide(100);

  //    var children_array = jQuery.makeArray(children);
  //    selected_ids = [];
  //    for (i = 0; i < children_array.length; i++) {
  //      selected_ids.push(children_array[i].children[0].id);
  //    }
  //    var children_string = selected_ids.join(', ');
  //    $.ajax({
  //      url: '/update_labels',
  //      data: jQuery.param({'image_id': children_string, 'species_id': species_id}),
  //      type: 'POST',
  //      success: function(response) {
  //        console.log("success " + response);
  //        // confirm that the number of rows updated in database matches the number of images selected
  //        var result = JSON.parse(response);
  //        var rows_updated = result['rows_updated'];
  //        if (children.length != rows_updated)
  //          alert(children.length + ' images were selected, but ' + rows_updated + ' were updated in the database!');

  //        for (i = 0; i < children.length; i++) {
  //          var child = children[i].children[1];
  //          child.innerHTML = species;
  //        }

  //        // ensure the message is on-screen by placing it under the cursor
  //        var image_str = " image";
  //        if (rows_updated > 1)
  //          image_str += "s";
  //        $('.toast').text(rows_updated + image_str + " labeled as " + species).fadeIn(250).delay(500).fadeOut(250).css({
  //          top: event.pageY + "px",
  //          left: event.pageX + "px",
  //        });
  //      },
  //      error: function(error) {
  //        console.log("error " + error);
  //      }
  //    });
  //});
});
