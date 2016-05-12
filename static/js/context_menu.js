$(document).on('keydown', function(e) {
  // F1
  if (e.which == 112) {
    e.preventDefault();
    //console.log('F1');
    repeat_last();
    return false;
  }
});

$(document).on('click', '#repeat-button', function() {
  //console.log('repeat button clicked');
  repeat_last();
});

$(document).on('click', '#junk-button', function() {
  //console.log('junk button clicked');
  junk_images();
});

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

function junk_images() {
  var children = $('#selectable').children('.ui-selected');

  var children_array = jQuery.makeArray(children);
  selected_ids = [];
  for (i = 0; i < children_array.length; i++) {
    selected_ids.push(children_array[i].children[0].id);
  }
  var children_string = selected_ids.join(', ');
  $.ajax({
    url: '/junk_images',
    data: jQuery.param({
      'image_id': children_string, 
    }),
    type: 'POST',
    success: function(response) {
      //console.log("success " + response);
      var selectable = $('#selectable');
      selectable.children().remove('.ui-selected');
      console.log('removing junked images');
    },
    error: function(error) {
      //console.log("error " + error);
    }
  });
}

function repeat_last() {
  var repeat_family = $('#repeat-family');
  var repeat_genus = $('#repeat-genus');
  var repeat_species = $('#repeat-species');

  var family_id = repeat_family.data('id');
  var family_name = repeat_family.data('name');
  var genus_id = repeat_genus.data('id');
  var genus_name = repeat_genus.data('name');
  var species_id = repeat_species.data('id');
  var species_name = repeat_species.data('name');

  if (family_id == 'None' && family_name == 'None') {
    family_id = null;
    family_name = null;
    genus_id = null;
    genus_name = null;
    species_id = null;
    species_name = null;
  }
  else if (genus_id == 'None' && genus_name == 'None') {
    genus_id = null;
    genus_name = null;
    species_id = null;
    species_name = null;
  }
  else if (species_id == 'None' && species_name == 'None') {
    species_id = null;
    species_name = null;
  }

  update_labels({
    'top': '50%',
    'left': '50%',
    'transform': 'translate(-50%, -50%)'}, {
    'family_id': family_id,
    'family_name': family_name,
    'genus_id': genus_id,
    'genus_name': genus_name,
    'species_id': species_id,
    'species_name': species_name,
  });
}

function update_labels(message, selection) {
  var species_id = selection['species_id'];
  var species_name = selection['species_name'];
  var genus_id = selection['genus_id'];
  var genus_name = selection['genus_name'];
  var family_id = selection['family_id'];
  var family_name = selection['family_name'];

  var children = $('#selectable').children('.ui-selected');
  // hide it AFTER the action was triggered
  $("#family-menu").hide(100);

  var children_array = jQuery.makeArray(children);
  selected_ids = [];
  for (i = 0; i < children_array.length; i++) {
    selected_ids.push(children_array[i].children[0].id);
  }
  var children_string = selected_ids.join(', ');
  $.ajax({
    url: '/update_labels',
    data: jQuery.param({
      'species_id': species_id, 
      'species_name': species_name,
      'genus_id': genus_id, 
      'genus_name': genus_name,
      'family_id': family_id, 
      'family_name': family_name,
      'image_id': children_string, 
    }),
    type: 'POST',
    success: function(response) {
      //console.log("success " + response);
      // confirm that the number of rows updated in database matches the number of images selected
      var result = JSON.parse(response);
      var rows_updated = result['rows_updated'];
      //console.log('success: ' + rows_updated);
      if (children.length != rows_updated)
        alert(children.length + ' images were selected, but ' + rows_updated + ' were updated in the database!');

      // display the most specific name we have for each image
      var name;
      //console.log(species_name + " " + genus_name + " " + family_name);
      var repeat_family = $('#repeat-family');
      if (family_name != null) {
        repeat_family.text(family_name);
        repeat_family.data('id', family_id);
        repeat_family.data('name', family_name);
      }
      else {
        repeat_family.text('None');
        repeat_family.data('id', 'None');
        repeat_family.data('name', 'None');
      }

      var repeat_genus = $('#repeat-genus');
      if (genus_name != null) {
        repeat_genus.text(genus_name);
        repeat_genus.data('id', genus_id);
        repeat_genus.data('name', genus_name);
      }
      else {
        repeat_genus.text('None');
        repeat_genus.data('id', 'None');
        repeat_genus.data('name', 'None');
      }

      var repeat_species = $('#repeat-species');
      if (species_name != null) {
        repeat_species.text(species_name);
        repeat_species.data('id', species_id);
        repeat_species.data('name', species_name);
      }
      else {
        repeat_species.text('None');
        repeat_species.data('id', 'None');
        repeat_species.data('name', 'None');
      }

      if (species_name != null)
        name = species_name;
      else if (genus_name != null)
        name = genus_name;
      else if (family_name != null)
        name = family_name;
      else
        name = "unknown";

      for (i = 0; i < children.length; i++) {
        var child = children[i].children[1];
        child.innerHTML = name;
      }
      
      var image_str = " image";
      if (rows_updated > 1)
        image_str += "s";
        // ensure the message is on-screen by placing it under the cursor
        $('.toast').text(rows_updated + image_str + " labeled as " + name).fadeIn(250).delay(500).fadeOut(250).css({
          top: message['top'],
          left: message['left'],
          transform: message['transform'],
      });
    },
    error: function(error) {
      //console.log("error " + error);
    }
  });
}

$(document).ready(function() {
  $(function(){
    // trigger only when the direct descendant list item is clicked
    $('#species-menu > li').on('click', function(e) {
      var species_clicked = $(this);
      var genus_clicked = species_clicked.closest('#genus-menu > li');
      var family_clicked = genus_clicked.closest('#family-menu > li');
      
      var species_id = species_clicked.attr('id');
      var species_name = species_clicked.attr('data-action');
      var genus_id = genus_clicked.attr('id');
      var genus_name = genus_clicked.attr('data-action');
      var family_id = family_clicked.attr('id');
      var family_name = family_clicked.attr('data-action');

      // need to make these null explicitly, because the context menu
      // labels are not converted to json before being passed
      if (species_id == "None" && species_name == "None") {
        species_id = null;
        species_name = null;
      }
      if (genus_id == "None" && genus_name == "None") {
        genus_id = null;
        genus_name = null;
      }

      console.log(
        "species_id: " + species_id + ", species_name: " + species_name + ", " +
        "genus_id: " + genus_id + ", genus_name: " + genus_name + ", " +
        "family_id: " + family_id + ", family_name: " + family_name
      );

      e.stopPropagation();
      update_labels({
        'top': e.pageY + 'px',
        'left': e.pageX + 'px',
        'transform': ''
        }, {
        'species_id': species_id,
        'species_name': species_name,
        'genus_id': genus_id,
        'genus_name': genus_name,
        'family_id': family_id,
        'family_name': family_name,
      });
    });

    $('#genus-menu > li').on('click', function(e) {
      var genus_clicked = $(this);
      var family_clicked = genus_clicked.closest('#family-menu > li');

      var species_id = null;
      var species_name = null;
      var genus_id = genus_clicked.attr('id');
      var genus_name = genus_clicked.attr('data-action');
      var family_id = family_clicked.attr('id');
      var family_name = family_clicked.attr('data-action');

      if (genus_id == "None" && genus_name == "None") {
        genus_id = null;
        genus_name = null;
      }

      console.log(
        "species_id: " + species_id + ", species_name: " + species_name + ", " +
        "genus_id: " + genus_id + ", genus_name: " + genus_name + ", " +
        "family_id: " + family_id + ", family_name: " + family_name
      );

      e.stopPropagation();
      update_labels({
        'top': e.pageY + 'px',
        'left': e.pageX + 'px',
        'transform': ''}, {
        'species_id': species_id,
        'species_name': species_name,
        'genus_id': genus_id,
        'genus_name': genus_name,
        'family_id': family_id,
        'family_name': family_name,
      });
    });

    $('#family-menu > li').on('click', function(e) {
      var family_clicked = $(this);

      var species_id = null;
      var species_name = null;
      var genus_id = null;
      var genus_name = null;
      var family_id = family_clicked.attr('id');
      var family_name = family_clicked.attr('data-action');

      if (family_id == "None" && family_name == "None") {
        family_id = null;
        family_name = null;
      }

      console.log(
        "species_id: " + species_id + ", species_name: " + species_name + ", " +
        "genus_id: " + genus_id + ", genus_name: " + genus_name + ", " +
        "family_id: " + family_id + ", family_name: " + family_name
      );

      e.stopPropagation();
      update_labels({
        'top': e.pageY + 'px',
        'left': e.pageX + 'px',
        'transform': ''}, {
        'species_id': species_id,
        'species_name': species_name,
        'genus_id': genus_id,
        'genus_name': genus_name,
        'family_id': family_id,
        'family_name': family_name,
      });
    });
  });
});
