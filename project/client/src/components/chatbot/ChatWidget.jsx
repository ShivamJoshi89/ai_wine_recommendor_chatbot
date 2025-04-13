// client/src/components/chatbot/ChatWidget.jsx
import React, { useState } from 'react';
import { Box, IconButton } from '@mui/material';
import ChatBubbleIcon from '@mui/icons-material/ChatBubble';
import CloseIcon from '@mui/icons-material/Close';
import ChatWindow from './ChatWindow';

const ChatWidget = () => {
  const [open, setOpen] = useState(false);

  const toggleChat = () => setOpen((prev) => !prev);

  return (
    <Box
      sx={{
        position: 'fixed',
        bottom: 24,
        right: 24,
        zIndex: 9999,
      }}
    >
      {open ? (
        <Box sx={{ position: 'relative', width: 300, height: 400, mb: 2 }}>
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              bgcolor: 'background.default',
              boxShadow: 3,
              borderRadius: 2,
              overflow: 'hidden',
            }}
          >
            <ChatWindow />
          </Box>
          <IconButton
            onClick={toggleChat}
            sx={{ position: 'absolute', top: 4, right: 4, color: 'text.primary' }}
          >
            <CloseIcon />
          </IconButton>
        </Box>
      ) : (
        <IconButton
          onClick={toggleChat}
          sx={{
            bgcolor: 'primary.main',
            color: 'primary.contrastText',
            boxShadow: 3,
            '&:hover': { bgcolor: 'primary.dark' },
          }}
        >
          <ChatBubbleIcon />
        </IconButton>
      )}
    </Box>
  );
};

export default ChatWidget;
