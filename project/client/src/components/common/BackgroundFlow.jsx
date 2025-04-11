import React, { useEffect, useState } from 'react';
import { Box } from '@mui/material';
import { motion } from 'framer-motion';
import { getWines } from '../../services/wineService';

export default function BackgroundFlow() {
  const [bgUrl, setBgUrl] = useState('');

  useEffect(() => {
    // Fetch the top (highest-priced) wine’s image URL
    getWines({ limit: 1, page: 1 }).then((data) => {
      if (data.length && data[0].image_url) {
        setBgUrl(data[0].image_url);
      }
    });
  }, []);

  if (!bgUrl) return null;

  const bottleVariants = {
    initial: { x: '-100vw', opacity: 0 },
    animate: {
      x: 0,
      opacity: 1,
      rotate: [0, 5, -5, 0],
      transition: {
        // Slide from left properties
        x: { type: 'spring', stiffness: 50, damping: 20, duration: 1.5 },
        opacity: { duration: 1.5 },
        // Then teeter continuously
        rotate: { duration: 2, repeat: Infinity, ease: 'easeInOut' },
      },
    },
  };

  return (
    <>
      {/* Full-screen dimmer */}
      <Box
        sx={{
          position: 'fixed',
          inset: 0,
          backgroundColor: 'rgba(0,0,0,0.4)',
          zIndex: -1,
        }}
      />

      {/* Animated bottle sliding in and teetering */}
      <motion.div
        variants={bottleVariants}
        initial="initial"
        animate="animate"
        style={{
          position: 'absolute',
          top: '180px', // Adjust vertical position as needed
          left: '42%',  // Adjust horizontal position: shifts bottle a bit to the left
          width: '300px', // Increase width as desired
          height: '600px', // Increase height as desired
          backgroundImage: `url(${bgUrl})`,
          backgroundSize: 'contain',
          backgroundRepeat: 'no-repeat',
          backgroundPosition: 'center bottom',
          pointerEvents: 'none',
          zIndex: 0,
        }}
      />
    </>
  );
}
