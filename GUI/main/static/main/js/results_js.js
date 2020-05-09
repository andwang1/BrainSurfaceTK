$('#predict').click(function () {
    document.getElementById("predict").style.color = "red";
    console.log("create post is working!");
    var table_Exists = document.getElementById("result_table");

    if (table_Exists == null) {

        $("#loader-wheel").fadeIn();

        var objCells = document.getElementById("info_table").rows.item(1).cells;
        console.log(objCells.item(0).innerHTML);

        $.ajax({
            url: predict_url,
            type: "GET",
            data: {
                'file_url': surf_file_url
            },


            success: function (json) {
                $('#post-text').val('');
                display_results(json.pred);
                $("#loader-wheel").fadeOut();
            }
        });
    }
});

$('#segment').click(function () {

    $("#loader-wheel").fadeIn();

    document.getElementById("segment").style.color = "red";
    console.log("create post is working!");
    var objCells = document.getElementById("info_table").rows.item(1).cells;
    console.log(objCells.item(0).innerHTML);

    $.ajax({
        url: segment_url,
        type: "GET",
        data: {
            'file_url': surf_file_url
        },
        success: function (json) {
            $('#post-text').val('');
            console.log("success");

            document.querySelector(".brain_surface_container").innerHTML = "";
            brain_surface_library.build_brain_surf_window(json.segmented_file_path, '.brain_surface_container');
            $("#loader-wheel").fadeOut();

            $.ajax({
                url: remove_tmp_url,
                type: "GET",
                data: {
                    'tmp_file_url': json.segmented_file_path
                },
                success: function () {
                    console.log("successfully removed tmp");
                }
            });
        }
    });
});


function display_results(pred) {
    var table_Exists = document.getElementById("result_table");
    if (table_Exists == null) {
        // CREATE DYNAMIC TABLE.
        var table = document.createElement('table');
        // SET THE TABLE ID.
        // WE WOULD NEED THE ID TO TRAVERSE AND EXTRACT DATA FROM THE TABLE.
        table.setAttribute('id', 'result_table');
        var arrHead = new Array();
        arrHead = ['Predicted age'];
        var arrValue = new Array();
        arrValue.push([pred]);
        var tr = table.insertRow(-1);
        for (var h = 0; h < arrHead.length; h++) {
            var th = document.createElement('th');              // TABLE HEADER.
            th.innerHTML = arrHead[h];
            tr.appendChild(th);
        }
        for (var c = 0; c <= arrValue.length - 1; c++) {
            tr = table.insertRow(-1);
            for (var j = 0; j < arrHead.length; j++) {
                var td = document.createElement('td');          // TABLE DEFINITION.
                td = tr.insertCell(-1);
                td.innerHTML = arrValue[c][j];                  // ADD VALUES TO EACH CELL.
            }
        }
        // FINALLY ADD THE NEWLY CREATED TABLE AND BUTTON TO THE BODY.
        document.getElementById("prediction_container").appendChild(table);
    }
}