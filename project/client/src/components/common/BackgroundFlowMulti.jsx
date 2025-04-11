import React, { useEffect, useState } from 'react';
// import { Box } from '@mui/material';
import AnimatedBottle from './AnimatedBottle';
import { getWines } from '../../services/wineService';

export default function BackgroundFlowMulti() {
  const [bottleData, setBottleData] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    // Fetch a larger set of wines (e.g., 50) then sort descending by price.
    getWines({ limit: 50, page: 1 })
      .then((data) => {
        if (data && data.length) {
          // Sort descending by price (assumes price is numeric or convertible)
          const sortedWines = data.sort(
            (a, b) => parseFloat(b.price) - parseFloat(a.price)
          );
          // Take the top 20 wines
          setBottleData(sortedWines.slice(0, 20));
        }
      })
      .catch((err) => {
        console.error('Error fetching wines for background:', err);
      });
  }, []);

  // Total animation time per bottle in milliseconds (must match AnimatedBottle timing)
  const totalAnimationTime = 5000; // 5 seconds (adjust if you change the duration in AnimatedBottle)

  // Cycle through the bottle images after each full animation loop
  useEffect(() => {
    if (bottleData.length === 0) return;

    const interval = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % bottleData.length);
    }, totalAnimationTime);

    return () => clearInterval(interval);
  }, [bottleData]);

  if (bottleData.length === 0) return null;

  const currentBottle = bottleData[currentIndex];

  return (
    <>
      {/* Full-screen dimmer overlay
      <Box
        sx={{
          position: 'fixed',
          inset: 0,
          backgroundColor: 'rgba(0,0,0,0.4)',
          zIndex: -1,
        }}
      /> */}

      {/* Render only the current bottle.
          The key prop forces a remount for each new bottle image so that the animation plays completely. */}
      <AnimatedBottle
        key={currentIndex}
        imageUrl={currentBottle.image_url}
        customDelay={0}
      />
    </>
  );
}
