// client/src/components/chatbot/MessageBubble.jsx
import React from 'react';
import { Box, Typography } from '@mui/material';
import { motion } from 'framer-motion';

const MessageBubble = ({ sender, text }) => {
  const isUser = sender === 'user';
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Box
        sx={{
          display: 'flex',
          justifyContent: isUser ? 'flex-end' : 'flex-start',
          mb: 1,
        }}
      >
        <Box
          sx={{
            bgcolor: isUser ? 'primary.main' : 'background.paper',
            color: isUser ? 'primary.contrastText' : 'text.primary',
            p: 1.5,
            borderRadius: 2,
            maxWidth: '70%',
          }}
        >
          <Typography variant="body1">{text}</Typography>
        </Box>
      </Box>
    </motion.div>
  );
};

export default MessageBubble;
