
// const dorabotImages = ['images/dorabot.jpg'];
// const fedexImages = ['images/fedex_img.jpg'];
// const lotteImages = ['images/lotte_1.jpg', 'images/lotte_2.jpg', 'images/lotte_3.jpg','images/lotte_4.jpg'];
// const teaImages = ['images/tea1.jpg', 'images/tea2.jpg'];


// function showImages(containerId, imagesArray) {
//     const container = document.getElementById(containerId);


//     container.innerHTML = '';


//     imagesArray.forEach(src => {
//         const img = document.createElement('img');
//         img.src = src;
//         img.alt = 'Project Image';
//         img.className = 'gallery-image';
//         container.appendChild(img);
//     });


//     container.classList.add('show');
// }


// function showDorabotImages() {
//     showImages('dorabot-images', dorabotImages);
// }

// function showFedExImages() {
//     showImages('fedex-images', fedexImages);
// }

// function showLotteImages() {
//     showImages('lotte-images', lotteImages);
// }

// function showteaImages() {
//     showImages('tea-images', teaImages);
// }


// 定义每个项目的图片数组
const dorabotImages = ['images/dorabot.jpg'];
const fedexImages = ['images/fedex_img.jpg'];
const lotteImages = ['images/lotte_1.jpg', 'images/lotte_2.jpg', 'images/lotte_3.jpg', 'images/lotte_4.jpg'];
const teaImages = ['images/tea1.jpg', 'images/tea2.jpg'];

// 通用的图片展示函数
function showImages(containerId, imagesArray) {
    const container = document.getElementById(containerId);

    // 清除已存在的图片
    container.innerHTML = '';

    // 循环创建图片元素并添加到容器
    imagesArray.forEach(src => {
        const img = document.createElement('img');
        img.src = src;
        img.alt = 'Project Image';
        img.className = 'gallery-image';
        container.appendChild(img);
    });

    // 添加淡入显示效果
    container.classList.add('show');
}

// 各项目的显示函数
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

// 滚动动画 - Intersection Observer
document.addEventListener('DOMContentLoaded', () => {
    const fadeInElements = document.querySelectorAll('.fade-in');

    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target); // 停止观察
            }
        });
    }, { threshold: 0.1 });

    fadeInElements.forEach(el => observer.observe(el));
});
