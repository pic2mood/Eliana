
var dir = '../training/data/test_images/';
var images = [];
var filenames = [];
var ctr = 1;

$.ajax({
	url: dir,
	success: function(data) {

		//console.log(data)

		$(data).find('a:contains("jpg")').each(function() {
			
			var filename = this.href.replace(window.location, '');
			filename = filename.replace('playground_scripts/', dir);
			
			//$('body').append('<img src="' + filename + '">');
			images.push('<img src="' + filename + '">');
			console.log(filename);
			filenames.push(filename);

			if (filenames.length == 1)
				$('#img').html('<img src="' + filenames[0] + '">');
		});
	}
});

var textFile = null;

var loader = function(emotion) {

	var textbox = document.getElementById('textbox');

	$('#img').html(images[ctr]);
	$('strong').html(filenames[ctr]);

	console.log(textbox.value);


	textbox.value += '\n' + filenames[ctr] + '|' + emotion;
	ctr++;
}

var makeTextFile = function() {
	var textbox = document.getElementById('textbox');
	var data = new Blob([textbox.value], {type: 'text/plain'});

	if (textFile !== null) {
	  	window.URL.revokeObjectURL(textFile);
	}

	textFile = window.URL.createObjectURL(data);

	return textFile;
}

function createDownloader() {
	var link = document.getElementById('downloadlink');
	link.href = makeTextFile();
	link.style.display = 'block';
}
