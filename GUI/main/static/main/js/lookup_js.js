$(document).ready(function () {

    document.getElementById("dropdown-session-id").addEventListener('change', function () {
        const selected_index = document.getElementById("dropdown-session-id").selectedIndex;
        const checkbox = document.getElementById("display-mri");
        mask_mri[selected_index - 1] ? checkbox.removeAttribute("disabled") : checkbox.setAttribute("disabled", "disabled")
    });

    document.getElementById("look-up-btn").addEventListener("click", function () {
        $("#loader-wheel").fadeIn();
    }, false);

    // const instance = document.querySelectorAll('select');
    // M.FormSelect.init(instance, {"dropdownOptions": {"hover": true}});
});
