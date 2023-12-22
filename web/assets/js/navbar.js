// Navbar Toogle
const toggleBtn = document.querySelector(".toggle_btn");
const toggleBtnIcon = document.querySelector(".toggle_btn i");
const dropDownMenu = document.querySelector(".dropdown_menu");

toggleBtn.onclick = function () {
  dropDownMenu.classList.toggle("open");
  const isOpen = dropDownMenu.classList.contains("open");
  toggleBtnIcon.classList = isOpen ? "fa-solid fa-xmark" : "fa-solid fa-bars";
};

// Active page indicator
// Get the current page URL
const currentPageUrl = window.location.href;

// Get all the <a> elements inside the navbar
const links = document.querySelectorAll("nav a");

// Set the 'active' class based on the current page URL
links.forEach((link) => {
  if (link.href === currentPageUrl) {
    link.classList.add("active");
  } else {
    link.classList.remove("active");
  }
});

// Animate on scroll 
AOS.init({
  easing: 'ease-out-back',
   duration: 1000,
   delay: 200,
   once: false,
   mirror: true,
   disable: 'phone',
   
});


