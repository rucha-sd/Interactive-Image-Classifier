var loadFile = function(event) {
    console.log("Hello")
    document.getElementById("image_uploaded").style.display = "block"
    document.getElementById("predict-btn").style.display = "block"
	var image = document.getElementById('uploaded_image')
	image.src = URL.createObjectURL(event.target.files[0])
}

$(document).ready(function() {
                $(".concept-prob").fancyTable({
                  /* Column number for initial sorting*/
                   sortColumn:0,
                   /* Setting pagination or enabling */
                   pagination: true,
                   /* Rows per page kept for display */
                   perPage:3,
                   globalSearch:true
                   });
                             
            });

            $(document).ready(function() {
                $(".classify-prob").fancyTable({
                  /* Column number for initial sorting*/
                   sortColumn:0,
                   /* Setting pagination or enabling */
                   pagination: true,
                   /* Rows per page kept for display */
                   perPage:3,
                   globalSearch:true
                   });
                             
            });
