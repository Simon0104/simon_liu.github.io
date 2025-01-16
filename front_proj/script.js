// Define image arrays for each project
const dorabotImages = ['images/dorabot.jpg'];
const fedexImages = ['images/fedex_img.jpg'];
const lotteImages = ['images/lotte_1.jpg', 'images/lotte_2.jpg', 'images/lotte_3.jpg', 'images/lotte_4.jpg'];
const teaImages = ['images/tea1.jpg', 'images/tea2.jpg'];

// Generic function to display images
function showImages(containerId, imagesArray) {
    const container = document.getElementById(containerId);

    // Clear existing images
    container.innerHTML = '';

    // Loop through the image array and create image elements
    imagesArray.forEach(src => {
        const img = document.createElement('img');
        img.src = src;
        img.alt = 'Project Image';
        img.className = 'gallery-image';
        container.appendChild(img);
    });

    // Add fade-in effect
    container.classList.add('show');
}

// Functions to display images for each project
function showDorabotImages() {
    showImages('dorabot-images', dorabotImages);
}

function showFedExImages() {
    showImages('fedex-images', fedexImages);
}

function showLotteImages() {
    showImages('lotte-images', lotteImages);
}

function showteaImages() {
    showImages('tea-images', teaImages);
}

// Scroll animation using Intersection Observer
document.addEventListener('DOMContentLoaded', () => {
    const fadeInElements = document.querySelectorAll('.fade-in');

    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target); // Stop observing after animation triggers
            }
        });
    }, { threshold: 0.1 });

    fadeInElements.forEach(el => observer.observe(el));
});
