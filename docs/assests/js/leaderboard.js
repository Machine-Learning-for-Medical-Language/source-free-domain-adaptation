$(document).ready( function () {
    $('#negation').DataTable({
        "order": [[ 3, "desc" ]],
        "searching": false, 
        "paging": false, 
        "info": false
    });
    $('#timex').DataTable({
        "order": [[ 3, "desc" ]],
        "searching": false, 
        "paging": false, 
        "info": false
    });
} );
