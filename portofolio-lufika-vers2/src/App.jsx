import React, { useState, useEffect, useRef } from 'react';

// --- Kustom Hook untuk Animasi Scroll ---
// Hook ini akan mengamati elemen dan mengembalikan status 'isVisible'
// saat elemen masuk ke viewport, yang memicu animasi fade-in.
const useIntersectionObserver = (options) => {
  const [isVisible, setIsVisible] = useState(false);
  const domRef = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          // Berhenti mengamati setelah terlihat
          observer.unobserve(entry.target);
        }
      });
    }, options);

    const { current } = domRef;
    if (current) {
      observer.observe(current);
    }

    return () => {
      if (current) {
        observer.unobserve(current);
      }
    };
  }, [options]);

  return [domRef, isVisible];
};

// --- Komponen FadeInWrapper ---
// Komponen wrapper untuk menerapkan animasi fade-in
const FadeInSection = ({ children, slideFrom = 'bottom' }) => {
  const [ref, isVisible] = useIntersectionObserver({
    threshold: 0.1, // Memicu saat 10% terlihat
    triggerOnce: true,
  });

  // Tentukan kelas transformasi berdasarkan arah slide
  let transformClasses = '';
  switch (slideFrom) {
    case 'left':
      transformClasses = 'opacity-0 -translate-x-10';
      break;
    case 'right':
      transformClasses = 'opacity-0 translate-x-10';
      break;
    default: // 'bottom'
      transformClasses = 'opacity-0 translate-y-10';
  }

  return (
    <div
      ref={ref}
      className={`transition-all duration-1000 ease-out ${
        isVisible ? 'opacity-100 translate-x-0 translate-y-0' : transformClasses
      }`}
    >
      {children}
    </div>
  );
};


// ===== 1. HEADER =====
const Header = () => {
  return (
    <header className="bg-deep-black text-soft-beige p-6 md:p-8 sticky top-0 z-50">
      <div className="container mx-auto flex justify-between items-center max-w-7xl px-4">
        {/* Logo */}
        <div className="text-xl md:text-2xl font-extrabold uppercase tracking-widest">
          <a href="#">Portofolio</a>
        </div>
        
        {/* Navigasi Desktop */}
        <nav className="hidden md:flex space-x-8 text-sm uppercase tracking-wider">
          <a href="#about" className="hover:opacity-70 transition-opacity duration-300">About</a>
          <a href="#featured" className="hover:opacity-70 transition-opacity duration-300">Gallery</a>
          <a href="#projects" className="hover:opacity-70 transition-opacity duration-300">Project</a>
          <a href="#contact" className="hover:opacity-70 transition-opacity duration-300">Contact</a>
        </nav>
        
        {/* Tombol Menu Mobile (hanya ikon) */}
        <button className="md:hidden text-soft-beige text-2xl">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16m-7 6h7" />
          </svg>
        </button>
      </div>
    </header>
  );
};


// ===== 2. HERO SECTION =====
const Hero = () => {
  return (
    <section className="bg-soft-beige text-deep-black py-24 px-6 md:px-12">
      <div className="container mx-auto max-w-7xl">
        {/* Headline Besar */}
        <h1 className="text-7xl md:text-9xl font-extrabold uppercase text-center tracking-tighter leading-none">
          THE FUTURE,
        </h1>
        {/* Subheadline */}
        <h2 className="text-2xl md:text-3xl font-extrabold uppercase text-center tracking-wider mt-4">
          ONE LINE OF CODE AT A TIME
        </h2>

        {/* Layout 2 Kolom */}
        <div className="mt-16 md:mt-24 grid md:grid-cols-2 gap-8 md:gap-12 items-stretch">
          
          {/* Kolom Kiri: Kartu Teks */}
          <div className="bg-soft-beige border border-deep-black rounded-2xl p-8 flex flex-col justify-between shadow-sm">
            <p className="font-sans text-base leading-relaxed max-w-md">
              I’m Luvi Asakura, an AI and web developer who loves blending creativity with intelligence — crafting digital experiences that think, feel, and inspire. Every project I build is a mix of logic and emotion, designed to make technology feel more human.
            </p>
            <button className="mt-6 bg-deep-black text-soft-beige py-3 px-8 rounded-full uppercase text-sm font-bold tracking-wider hover:bg-dark-gray transition-colors duration-300 self-start">
              About Us
            </button>
          </div>

          {/* Kolom Kanan: Foto */}
          <div className="w-full h-full">
            <img 
              src="https://placehold.co/600x700/3a3a3a/e6d8c3?text=Photographer" 
              alt="Portrait of the photographer Amelia Allen"
              className="w-full h-full object-cover rounded-2xl shadow-lg"
              style={{ minHeight: '300px' }}
              onError={(e) => e.target.src='https://placehold.co/600x700/3a3a3a/e6d8c3?text=Image+Error'}
            />
          </div>
        </div>
      </div>
    </section>
  );
};


// ===== 3. ABOUT ME SECTION =====
const About = () => {
  return (
    <section id="about" className="bg-soft-beige text-deep-black py-24 px-6 md:px-12">
      <div className="container mx-auto max-w-7xl">
        
        {/* Grid: Judul di kiri, Teks di kanan */}
        <div className="grid md:grid-cols-2 gap-8 md:gap-12 items-start">
          <h2 className="text-5xl md:text-6xl font-extrabold uppercase tracking-tighter">
            About Me
          </h2>
          <p className="font-sans text-base lowercase leading-relaxed text-dark-gray max-w-lg">
            Starting her journey in web design and artificial intelligence at 19, Luvi Asakura quickly discovered her passion for creating technology that feels human. Her work blends logic and emotion, turning complex systems into meaningful digital experiences. With a deep curiosity for innovation, she continues to explore how design and AI can shape the future of interaction.
          </p>
        </div>

        {/* 3 Foto Horizontal di bawah */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8">
          {[
            { src: "https://i.pinimg.com/1200x/0d/b4/91/0db491bb75e4282766d05546aefd6cb8.jpg", alt: "AI Bot - MindEase (Mindy)" },
            { src: "https://i.pinimg.com/736x/83/23/6d/83236d2f560e6e8938fec2238f83cefe.jpg", alt: "Music stample" },
            { src: "https://i.pinimg.com/736x/a8/8a/8e/a88a8e8b5b4544d8a34b264841ba8956.jpg", alt: "tamplate-project" },
          ].map((img, index) => (
            <div key={index} className="overflow-hidden rounded-2xl shadow-md group">
              <img 
                src={img.src} 
                alt={img.alt}
                className="w-full h-full object-cover transition-transform duration-300 ease-in-out group-hover:scale-105"
                onError={(e) => e.target.src=`https://placehold.co/500x400/aaaaaa/1c1b1b?text=Image+${index+1}`}
              />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};


// ===== 4. FEATURED WORKS SECTION =====
const FeaturedItem = ({ title, year, imgSrc, slideFrom }) => {
  return (
    <FadeInSection slideFrom={slideFrom}>
      <a href="#" className="block relative h-56 md:h-64 rounded-2xl overflow-hidden group shadow-lg">
        {/* Gambar Background */}
        <img 
          src={imgSrc} 
          alt={title} 
          className="absolute inset-0 w-full h-full object-cover transition-transform duration-500 ease-in-out group-hover:scale-110"
          onError={(e) => e.target.src='https://placehold.co/1200x400/555555/ffffff?text=Image+Error'}
        />
        {/* Overlay Gelap */}
        <div className="absolute inset-0 bg-black bg-opacity-50 group-hover:bg-opacity-40 transition-all duration-300"></div>
        {/* Teks */}
        <div className="absolute inset-0 flex items-center justify-center p-6">
          <h3 className="text-3xl md:text-5xl font-extrabold uppercase text-white tracking-wider text-center">
            {title} – {year}
          </h3>
        </div>
      </a>
    </FadeInSection>
  );
};

const Featured = () => {
  const works = [
    { title: "Art Director", year: "2021", imgSrc: "https://placehold.co/1200x400/555555/ffffff?text=Project+One", slideFrom: "left" },
    { title: "Photographer", year: "2021", imgSrc: "https://placehold.co/1200x400/666666/ffffff?text=Project+Two", slideFrom: "right" },
    { title: "Videographer", year: "2022", imgSrc: "https://placehold.co/1200x400/777777/ffffff?text=Project+Three", slideFrom: "left" },
  ];

  return (
    <section id="featured" className="bg-deep-black text-soft-beige py-24 px-6 md:px-12">
      <div className="container mx-auto max-w-7xl">
        <FadeInSection>
          <h2 className="text-5xl md:text-6xl font-extrabold uppercase tracking-tighter mb-16">
            Featured Works
          </h2>
        </FadeInSection>
        
        {/* Tumpukan Banner */}
        <div className="space-y-8">
          {works.map((work, index) => (
            <FeaturedItem 
              key={index} 
              title={work.title} 
              year={work.year} 
              imgSrc={work.imgSrc}
              slideFrom={work.slideFrom}
            />
          ))}
        </div>
      </div>
    </section>
  );
};


// ===== 5. PROJECTS SECTION =====
const Projects = () => {
  // Daftar 5 gambar untuk grid
  const projectImages = [
    // Baris Atas (3 gambar persegi)
    { src: "https://placehold.co/500x500/d8c8b3/1c1b1b?text=People", alt: "People portrait", className: "col-span-2 row-span-1 aspect-square" },
    { src: "https://placehold.co/500x500/c9b9a9/1c1b1b?text=Outdoors", alt: "Outdoor photography", className: "col-span-2 row-span-1 aspect-square" },
    { src: "https://placehold.co/500x500/bfae9e/1c1b1b?text=Creative", alt: "Creative shot", className: "col-span-2 row-span-1 aspect-square" },
    // Baris Bawah (2 gambar lebar)
    { src: "https://placehold.co/600x400/b5a494/1c1b1b?text=Product", alt: "Product shot", className: "col-span-3 row-span-1 aspect-video md:aspect-[16/9]" },
    { src: "https://placehold.co/600x400/ac9a89/1c1b1b?text=Sculpture", alt: "Sculpture art", className: "col-span-3 row-span-1 aspect-video md:aspect-[16/9]" },
  ];

  return (
    <section id="projects" className="bg-soft-beige text-deep-black py-24 px-6 md:px-12">
      <div className="container mx-auto max-w-7xl">
        
        {/* Header Bagian Proyek */}
        <div className="flex justify-between items-center mb-12">
          <h2 className="text-5xl md:text-6xl font-extrabold uppercase tracking-tighter">
            Projects
          </h2>
          <a href="#" className="text-xs md:text-sm uppercase font-bold tracking-widest border border-deep-black px-5 py-3 rounded-full hover:bg-deep-black hover:text-soft-beige transition-colors duration-300 whitespace-nowrap">
            Show All
          </a>
        </div>
        
        {/* Grid Proyek (3 atas, 2 bawah) */}
        {/* Menggunakan grid 6-kolom untuk fleksibilitas */}
        <div className="grid grid-cols-6 grid-rows-2 gap-6 md:gap-8">
          {projectImages.map((img, index) => (
            <div 
              key={index} 
              className={`group relative overflow-hidden rounded-2xl shadow-md ${img.className}`}
            >
              <img 
                src={img.src} 
                alt={img.alt} 
                className="w-full h-full object-cover transition-transform duration-300 ease-in-out group-hover:scale-105"
                onError={(e) => e.target.src=`https://placehold.co/500x500/aaaaaa/1c1b1b?text=Image+${index+1}`}
              />
              <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-10 transition-opacity duration-300"></div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};


// ===== 6. FOOTER =====
const Footer = () => {
  return (
    <footer id="contact" className="bg-soft-beige text-deep-black pt-24">
      {/* Bagian Atas Footer (Kontak) */}
      <FadeInSection>
        <div className="container mx-auto max-w-7xl px-6 md:px-12">
          <div className="flex flex-col md:flex-row justify-between items-center gap-8 md:gap-12 text-center md:text-left">
            {/* Email */}
            <a href="mailto:hello@gmail.com" className="text-2xl md:text-3xl font-light hover:opacity-70 transition-opacity duration-300">
              Say hello ✈ hello@gmail.com
            </a>
            
            {/* Sosial Media */}
            <div className="flex items-center space-x-3 md:space-x-4 font-medium uppercase tracking-wider text-sm">
              <a href="#" className="hover:opacity-70 transition-opacity duration-300">LinkedIn</a>
              <span>|</span>
              <a href="#" className="hover:opacity-70 transition-opacity duration-300">Instagram</a>
              <span>|</span>
              <a href="#" className="hover:opacity-70 transition-opacity duration-300">Twitter</a>
            </div>
          </div>
        </div>
      </FadeInSection>
      
      {/* Bagian Bawah Footer (Logo Besar) */}
      <div className="mt-24 bg-deep-black text-soft-beige py-16 md:py-20 text-center">
        <h3 className="text-6xl md:text-9xl font-extrabold uppercase tracking-tighter opacity-90">
          Photography
        </h3>
      </div>
    </footer>
  );
};


// ===== KOMPONEN UTAMA APP =====
export default function App() {
  return (
    // Gunakan warna-warna kustom dari config
    // Kita asumsikan tailwind.config.js sudah disetup,
    // tapi untuk file tunggal, kita akan gunakan style inline untuk ini.
    // Namun, kelas `bg-deep-black` dll akan berfungsi jika disuntikkan.
    // Mari kita tambahkan style global untuk warna.
    <>
      <style>{`
        body {
          font-family: 'Inter', sans-serif; /* Fallback font */
        }
        h1, h2, h3, h4, h5, h6, .font-extrabold {
          font-family: 'Poppins', sans-serif; /* Fallback font */
          font-weight: 800; /* ExtraBold */
        }
        
        /* Definisi Warna Kustom Tailwind */
        :root {
          --color-deep-black: #1c1b1b;
          --color-soft-beige: #e6d8c3;
          --color-dark-gray: #2a2a2a;
        }
        .bg-deep-black { background-color: var(--color-deep-black); }
        .bg-soft-beige { background-color: var(--color-soft-beige); }
        .text-deep-black { color: var(--color-deep-black); }
        .text-soft-beige { color: var(--color-soft-beige); }
        .text-dark-gray { color: var(--color-dark-gray); }
        .border-deep-black { border-color: var(--color-deep-black); }
        
        /* Google Fonts Import (jika React dimuat di HTML dengan ini) */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;800;900&family=Inter:wght@400;500&display=swap');
      `}</style>
      
      <div className="bg-soft-beige antialiased">
        <Header />
        <main>
          <FadeInSection>
            <Hero />
          </FadeInSection>
          <FadeInSection>
            <About />
          </FadeInSection>
          
          {/* Featured memiliki animasi internal per item */}
          <Featured /> 
          
          <FadeInSection>
            <Projects />
          </FadeInSection>
        </main>
        {/* Footer memiliki animasi internal di bagian atasnya */}
        <Footer /> 
      </div>
    </>
  );
}
